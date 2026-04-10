#!/usr/bin/env python3
"""
Scan all parasol benchmark results for data quality issues.

Walks parasol-benchmarks/, treats every directory containing a results.csv as a
"config", and groups configs by their parent directory ("experiment").

For each config it checks:
  - Status / time inconsistencies (Unknown but exited early, Optimal but no
    objective, time = 0, etc.)
  - Time field parsing issues (empty / non-numeric / negative / runaway)
  - Missing or empty .out files
  - Crash markers in .out files
  - Schedule column inconsistency
  - Duplicate (problem, name) rows

After per-config checks it does cross-config checks within each experiment:
  - Different instance sets across configs
  - Same instance Optimal in two configs but with different objectives

Run with --verbose to see every offending row; default output is a summary.
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "parasol-benchmarks"
SCORING_DIR = ROOT.parent / "scoring"
TYPES_CSV = SCORING_DIR / "problem_types.csv"

# Timeout in ms. Runs can overshoot slightly due to measurement, so we treat
# anything above EARLY_EXIT_THRESHOLD as "ran the full timeout".
TIMEOUT_MS = 1_200_000
EARLY_EXIT_THRESHOLD = 1_150_000   # below this, an Unknown row is suspicious
RUNAWAY_THRESHOLD     = 1_250_000  # above this, the time is unreasonably long
TINY_OPTIMAL_THRESHOLD = 100        # ms — Optimal in less than this is suspicious

CRASH_PATTERNS = re.compile(
    r"(Traceback|Segmentation fault|Killed|Aborted|panicked|MemoryError|"
    r"std::bad_alloc|core dumped|OutOfMemoryError)",
    re.IGNORECASE,
)


def load_problem_types() -> dict[str, str]:
    types = {}
    if not TYPES_CSV.exists():
        return types
    with open(TYPES_CSV) as f:
        for r in csv.DictReader(f):
            types[r["model"]] = r["type"]
    return types


def discover_configs(root: Path) -> dict[Path, list[Path]]:
    """Return {experiment_dir: [config_dir, ...]} for every results.csv found."""
    configs = sorted(p.parent for p in root.rglob("results.csv"))
    experiments: dict[Path, list[Path]] = defaultdict(list)
    for c in configs:
        experiments[c.parent].append(c)
    return experiments


def parse_time(value: str) -> tuple[int | None, str | None]:
    """Return (time_ms, error) — error is None on success."""
    if value == "" or value is None:
        return None, "empty"
    try:
        t = int(value)
    except ValueError:
        return None, f"non-numeric ({value!r})"
    if t < 0:
        return None, f"negative ({t})"
    return t, None


def out_file_for(config_dir: Path, row: dict) -> Path | None:
    """Find a matching .out file for a row, ignoring schedule/cores/trial."""
    pattern = f"{row['problem']}-sep-{row['model']}-sep-{row['name']}-sep-*.out"
    matches = list(config_dir.glob(pattern))
    if not matches:
        return None
    # If multiple, take the most recently modified
    return max(matches, key=lambda p: p.stat().st_mtime)


def check_config(config_dir: Path, problem_types: dict[str, str], verbose: bool):
    """Run all checks on one config dir. Returns dict of issue counts."""
    csv_path = config_dir / "results.csv"
    issues: dict[str, list[str]] = defaultdict(list)

    rows = list(csv.DictReader(open(csv_path)))
    issues["__rows__"] = [str(len(rows))]

    if not rows:
        issues["empty_csv"].append(str(csv_path))
        return issues

    # Schedule consistency
    schedules = {r["schedule"] for r in rows}
    if len(schedules) > 1:
        issues["schedule_inconsistent"].append(f"schedules in CSV: {sorted(schedules)}")

    # Duplicates
    seen: dict[tuple, int] = defaultdict(int)
    for r in rows:
        seen[(r["problem"], r["name"])] += 1
    dups = [k for k, v in seen.items() if v > 1]
    for d in dups:
        issues["duplicate_row"].append(f"{d[0]}/{d[1]}")

    for r in rows:
        key = f"{r['problem']}/{r['name']}"
        kind = problem_types.get(r["model"])  # may be None — we'll note it
        status = r["optimal"]
        objective = r["objective"]

        # Time parsing
        t, err = parse_time(r["time_ms"])
        if err is not None:
            issues["time_parse_error"].append(f"{key}: {err}")
            t = None

        # Time bounds
        if t is not None:
            if t > RUNAWAY_THRESHOLD:
                issues["time_runaway"].append(f"{key}: {t}ms")
            if t == 0:
                issues["time_zero"].append(key)

        # Status / time inconsistencies
        if status == "Unknown" and t is not None and t < EARLY_EXIT_THRESHOLD:
            issues["unknown_early_exit"].append(f"{key}: {t}ms")
        if status == "Optimal" and t is not None and t < TINY_OPTIMAL_THRESHOLD:
            issues["optimal_tiny_time"].append(f"{key}: {t}ms")

        # Optimal but no objective (only meaningful for OPT problems)
        if kind in ("MIN", "MAX") and status == "Optimal" and objective == "":
            issues["optimal_no_objective"].append(key)

        # Unknown problem type — can't check semantics
        if kind is None:
            issues["unknown_model_type"].append(f"{key} (model={r['model']})")

        # .out file checks
        out = out_file_for(config_dir, r)
        if out is None:
            issues["out_missing"].append(key)
        else:
            try:
                size = out.stat().st_size
            except OSError:
                size = 0
            if size == 0:
                issues["out_empty"].append(f"{key} -> {out.name}")
            else:
                # Read tail to scan for crash markers (cheap)
                try:
                    with open(out, "rb") as f:
                        f.seek(max(0, size - 8192))
                        tail = f.read().decode("utf-8", errors="replace")
                except OSError:
                    tail = ""
                if CRASH_PATTERNS.search(tail):
                    m = CRASH_PATTERNS.search(tail)
                    issues["out_crash_marker"].append(f"{key}: '{m.group(0)}'")

    return issues


def cross_config_checks(experiment: Path, config_dirs: list[Path],
                        problem_types: dict[str, str]) -> dict[str, list[str]]:
    """Compare configs within an experiment for cross-cutting issues."""
    issues: dict[str, list[str]] = defaultdict(list)

    # Load all
    config_rows: dict[str, dict[tuple, dict]] = {}
    for c in config_dirs:
        rows = {}
        with open(c / "results.csv") as f:
            for r in csv.DictReader(f):
                rows[(r["problem"], r["name"])] = r
        config_rows[c.name] = rows

    # Instance set differences
    all_keys = set().union(*[set(d.keys()) for d in config_rows.values()])
    for name, rows in config_rows.items():
        missing = all_keys - set(rows.keys())
        if missing:
            issues["instance_set_mismatch"].append(
                f"{name} missing {len(missing)} instances (e.g. {sorted(missing)[:3]})"
            )

    # Conflicting Optimal objectives
    for k in all_keys:
        opts = []
        for name, rows in config_rows.items():
            r = rows.get(k)
            if r is None:
                continue
            if r["optimal"] == "Optimal" and r["objective"] != "":
                opts.append((name, r["objective"]))
        if len(opts) >= 2:
            objs = {o[1] for o in opts}
            if len(objs) > 1:
                model = next(rows[k]["model"] for rows in config_rows.values() if k in rows)
                kind = problem_types.get(model, "?")
                issues["optimal_conflicting_objectives"].append(
                    f"{k[0]}/{k[1]} ({kind}): " +
                    ", ".join(f"{n}={o}" for n, o in opts)
                )

    return issues


def print_summary(label: str, issues: dict[str, list[str]], verbose: bool, indent: str = ""):
    rows_count = issues.pop("__rows__", ["?"])[0] if "__rows__" in issues else None
    cats = sorted((k, v) for k, v in issues.items() if v)
    header = f"{indent}{label}"
    if rows_count is not None:
        header += f"  ({rows_count} rows)"
    if not cats:
        print(f"{header}  ✔ clean")
        return
    print(header)
    for cat, items in cats:
        print(f"{indent}  • {cat}: {len(items)}")
        if verbose:
            for it in items[:20]:
                print(f"{indent}      - {it}")
            if len(items) > 20:
                print(f"{indent}      ... +{len(items) - 20} more")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="show offending rows under each issue")
    ap.add_argument("--filter", "-f", default=None,
                    help="only scan experiments whose name contains this substring")
    args = ap.parse_args()

    problem_types = load_problem_types()
    if not problem_types:
        print(f"WARNING: no problem types loaded from {TYPES_CSV}", file=sys.stderr)

    experiments = discover_configs(DATA_ROOT)
    if not experiments:
        print(f"No results.csv files found under {DATA_ROOT}", file=sys.stderr)
        sys.exit(1)

    grand_totals: dict[str, int] = defaultdict(int)

    for exp_dir in sorted(experiments):
        if args.filter and args.filter not in exp_dir.name:
            continue
        configs = experiments[exp_dir]
        rel = exp_dir.relative_to(DATA_ROOT)
        print(f"\n═══ {rel} ═══")
        for c in sorted(configs):
            issues = check_config(c, problem_types, args.verbose)
            for k, v in issues.items():
                if k != "__rows__":
                    grand_totals[k] += len(v)
            print_summary(c.name, issues, args.verbose, indent="  ")

        cross = cross_config_checks(exp_dir, configs, problem_types)
        if cross:
            print(f"  -- cross-config --")
            for k, v in cross.items():
                grand_totals[k] += len(v)
            print_summary("(cross)", {"__rows__": [str(sum(len(v) for v in cross.values()))], **cross},
                          args.verbose, indent="  ")

    print("\n═══ GRAND TOTALS ═══")
    if not grand_totals:
        print("No issues found.")
    else:
        for k in sorted(grand_totals):
            print(f"  {k}: {grand_totals[k]}")


if __name__ == "__main__":
    main()
