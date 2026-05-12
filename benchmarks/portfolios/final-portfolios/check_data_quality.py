#!/usr/bin/env python3
"""
Scan portfolios/final-portfolios benchmark results for data quality issues.

Layout assumed (test-orchestrator):
    {data_root}/{portfolio}/{portfolio}-{year}-r{rep}/results.csv

The legacy layout (no rep suffix) is also supported when --data-root points
at portfolios-final/.

Per-rep checks (same as before):
  - Status / time inconsistencies (Unknown but exited early, Optimal but no
    objective, time = 0, etc.)
  - Time field parsing issues (empty / non-numeric / negative / runaway)
  - Missing or empty .out files
  - Crash markers in .out files
  - Crash in .out but status is Optimal
  - Optimal but .out has no ========== (search-complete marker)
  - CSV objective doesn't match last _objective in .out file
  - Schedule column inconsistency
  - Duplicate (problem, name) rows

Cross-rep checks (within a single (portfolio, year)):
  - Wrong number of reps (expected EXPECTED_REPS)
  - Different instance sets across reps
  - Reps disagree on status for the same instance
  - Reps disagree on objective when both proved Optimal
  - Reps' times disagree sharply (max/min ratio above TIME_DISAGREE_RATIO)

Run with --verbose to see every offending row; default output is a summary.
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = ROOT / "test-orchestrator"
TYPES_CSV = ROOT.parent.parent / "open-category-benchmarks" / "problem_types.csv"

TIMEOUT_MS = 1_200_000
EARLY_EXIT_THRESHOLD = 1_150_000
RUNAWAY_THRESHOLD = 1_250_000
TINY_OPTIMAL_THRESHOLD = 100
TIME_DISAGREE_RATIO = 5.0
EXPECTED_REPS = 3

# Match {portfolio}-{year}-r{rep}; rep is optional for legacy support.
DIR_RE = re.compile(r"^(?P<portfolio>.+)-(?P<year>\d{4})(?:-r(?P<rep>\d+))?$")

CRASH_PATTERNS = re.compile(
    r"(Traceback|Segmentation fault|Killed|Aborted|panicked|MemoryError|"
    r"std::bad_alloc|core dumped|OutOfMemoryError)",
    re.IGNORECASE,
)
OBJECTIVE_RE = re.compile(r"^_objective\s*=\s*(-?\d+)\s*;", re.MULTILINE)


def load_problem_types() -> dict[str, str]:
    types = {}
    if not TYPES_CSV.exists():
        return types
    with open(TYPES_CSV) as f:
        for r in csv.DictReader(f):
            types[r["model"]] = r["type"]
    return types


def parse_dir_name(name: str) -> tuple[str, str, str | None] | None:
    m = DIR_RE.match(name)
    if not m:
        return None
    return m.group("portfolio"), m.group("year"), m.group("rep")


def discover_runs(root: Path):
    """Yield (portfolio, year, rep_or_None, run_dir) for each results.csv."""
    for csv_path in sorted(root.rglob("results.csv")):
        run_dir = csv_path.parent
        parsed = parse_dir_name(run_dir.name)
        if parsed is None:
            print(f"WARN: skipping {run_dir} (name doesn't match {{portfolio}}-{{year}}[-r{{rep}}])",
                  file=sys.stderr)
            continue
        yield (*parsed, run_dir)


def parse_time(value: str) -> tuple[int | None, str | None]:
    if value == "" or value is None:
        return None, "empty"
    try:
        t = int(value)
    except ValueError:
        return None, f"non-numeric ({value!r})"
    if t < 0:
        return None, f"negative ({t})"
    return t, None


def out_file_for(run_dir: Path, row: dict) -> Path | None:
    pattern = f"{row['problem']}-sep-{row['model']}-sep-{row['name']}-sep-*.out"
    matches = list(run_dir.glob(pattern))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def check_run(run_dir: Path, problem_types: dict[str, str]):
    csv_path = run_dir / "results.csv"
    issues: dict[str, list[str]] = defaultdict(list)

    rows = list(csv.DictReader(open(csv_path)))
    issues["__rows__"] = [str(len(rows))]

    if not rows:
        issues["empty_csv"].append(str(csv_path))
        return issues, rows

    schedules = {r["schedule"] for r in rows}
    if len(schedules) > 1:
        issues["schedule_inconsistent"].append(f"schedules in CSV: {sorted(schedules)}")

    seen: dict[tuple, int] = defaultdict(int)
    for r in rows:
        seen[(r["problem"], r["name"])] += 1
    for k, v in seen.items():
        if v > 1:
            issues["duplicate_row"].append(f"{k[0]}/{k[1]}")

    for r in rows:
        key = f"{r['problem']}/{r['name']}"
        kind = problem_types.get(r["model"])
        status = r["optimal"]
        objective = r["objective"]

        t, err = parse_time(r["time_ms"])
        if err is not None:
            issues["time_parse_error"].append(f"{key}: {err}")
            t = None

        if t is not None:
            if t > RUNAWAY_THRESHOLD:
                issues["time_runaway"].append(f"{key}: {t}ms")
            if t == 0:
                issues["time_zero"].append(key)

        if status == "Unknown" and t is not None and t < EARLY_EXIT_THRESHOLD:
            issues["unknown_early_exit"].append(f"{key}: {t}ms")
        if status == "Optimal" and t is not None and t < TINY_OPTIMAL_THRESHOLD:
            issues["optimal_tiny_time"].append(f"{key}: {t}ms")

        if kind in ("MIN", "MAX") and status == "Optimal" and objective == "":
            issues["optimal_no_objective"].append(key)

        if kind is None:
            issues["unknown_model_type"].append(f"{key} (model={r['model']})")

        out = out_file_for(run_dir, r)
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
                try:
                    content = open(out, "r", errors="replace").read()
                except OSError:
                    content = ""

                has_crash = CRASH_PATTERNS.search(content)
                has_solution = "----------" in content
                has_complete = "==========" in content

                if has_crash:
                    issues["out_crash_marker"].append(f"{key}: '{has_crash.group(0)}'")
                    if status == "Optimal":
                        issues["crash_but_optimal"].append(key)

                if status == "Optimal" and not has_complete and kind in ("MIN", "MAX"):
                    issues["optimal_no_complete_marker"].append(key)

                if objective and has_solution:
                    obj_matches = OBJECTIVE_RE.findall(content)
                    if obj_matches:
                        last_obj = obj_matches[-1]
                        if last_obj != objective:
                            issues["objective_mismatch"].append(
                                f"{key}: csv={objective} out={last_obj}")

    return issues, rows


def cross_rep_checks(year_label: str, rep_runs: dict[str, list[dict]],
                     problem_types: dict[str, str]) -> dict[str, list[str]]:
    """Compare reps within a single (portfolio, year) group."""
    issues: dict[str, list[str]] = defaultdict(list)

    if len(rep_runs) != EXPECTED_REPS:
        issues["wrong_rep_count"].append(
            f"{year_label}: have {len(rep_runs)} reps {sorted(rep_runs.keys())}, "
            f"expected {EXPECTED_REPS}"
        )

    rep_rows: dict[str, dict[tuple, dict]] = {}
    for rep, rows in rep_runs.items():
        rep_rows[rep] = {(r["problem"], r["name"]): r for r in rows}

    all_keys = set().union(*[set(d.keys()) for d in rep_rows.values()]) if rep_rows else set()
    for rep, rows in rep_rows.items():
        missing = all_keys - set(rows.keys())
        if missing:
            issues["instance_set_mismatch"].append(
                f"rep={rep} missing {len(missing)} (e.g. {sorted(missing)[:3]})"
            )

    for k in all_keys:
        present = {rep: rows[k] for rep, rows in rep_rows.items() if k in rows}
        statuses = {rep: r["optimal"] for rep, r in present.items()}
        objectives = {rep: r["objective"] for rep, r in present.items() if r["objective"]}
        times = {}
        for rep, r in present.items():
            t, _ = parse_time(r["time_ms"])
            if t is not None:
                times[rep] = t

        if len(set(statuses.values())) > 1:
            issues["rep_status_disagreement"].append(
                f"{k[0]}/{k[1]}: " + ", ".join(f"{rep}={s}" for rep, s in sorted(statuses.items()))
            )

        opt_objs = {rep: r["objective"] for rep, r in present.items()
                    if r["optimal"] == "Optimal" and r["objective"]}
        if len(set(opt_objs.values())) > 1:
            issues["rep_optimal_objective_disagreement"].append(
                f"{k[0]}/{k[1]}: " + ", ".join(f"{rep}={o}" for rep, o in sorted(opt_objs.items()))
            )

        if len(times) >= 2:
            tmin, tmax = min(times.values()), max(times.values())
            if tmin > 0 and tmax / tmin >= TIME_DISAGREE_RATIO:
                issues["rep_time_disagreement"].append(
                    f"{k[0]}/{k[1]}: " +
                    ", ".join(f"{rep}={t}ms" for rep, t in sorted(times.items()))
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
                    help="only scan portfolios whose name contains this substring")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                    help=f"data root (default: {DEFAULT_DATA_ROOT.relative_to(ROOT)})")
    args = ap.parse_args()

    problem_types = load_problem_types()
    if not problem_types:
        print(f"WARNING: no problem types loaded from {TYPES_CSV}", file=sys.stderr)

    runs = list(discover_runs(args.data_root))
    if not runs:
        print(f"No results.csv files found under {args.data_root}", file=sys.stderr)
        sys.exit(1)

    # Group: {portfolio: {year: {rep: rows}}}
    grouped: dict[str, dict[str, dict[str, list[dict]]]] = defaultdict(lambda: defaultdict(dict))
    grand_totals: dict[str, int] = defaultdict(int)

    for portfolio, year, rep, run_dir in runs:
        if args.filter and args.filter not in portfolio:
            continue
        rep_key = rep if rep is not None else "_"
        issues, rows = check_run(run_dir, problem_types)
        for k, v in issues.items():
            if k != "__rows__":
                grand_totals[k] += len(v)
        label = f"{portfolio}-{year}" + (f"-r{rep}" if rep else "")
        print_summary(label, issues, args.verbose, indent="  ")
        grouped[portfolio][year][rep_key] = rows

    print()
    for portfolio in sorted(grouped):
        print(f"\n═══ cross-rep: {portfolio} ═══")
        for year in sorted(grouped[portfolio]):
            label = f"{portfolio}-{year}"
            cross = cross_rep_checks(label, grouped[portfolio][year], problem_types)
            for k, v in cross.items():
                grand_totals[k] += len(v)
            print_summary(label, cross, args.verbose, indent="  ")

    print("\n═══ GRAND TOTALS ═══")
    if not grand_totals:
        print("No issues found.")
    else:
        for k in sorted(grand_totals):
            print(f"  {k}: {grand_totals[k]}")


if __name__ == "__main__":
    main()
