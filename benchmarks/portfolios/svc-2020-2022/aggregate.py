#!/usr/bin/env python3
"""
Aggregate the per-rep test-orchestrator results into two CSVs:

  combined.csv         one row per (portfolio, year, rep, instance)
  combined_median.csv  one row per (portfolio, year, instance); the
                       median rep's (status, objective, time_ms) verbatim

Median rep selection:
  For each (portfolio, year, instance) group, the reps play a pairwise
  Borda tournament against each other using utils.borda._compare. The rep
  with the middle Borda score is the canonical rep; its status, objective
  and time_ms are copied into the aggregated row.

  _compare orders by: solved > unsolved, then better objective (MIN/MAX),
  then complete > incomplete on equal objectives, then lower time. So the
  median rep is the one that's "neither best nor worst" by that ordering.

  Ties on Borda score are resolved by rep number (stable sort, r1 < r2 < r3).

Status translation: results.csv uses {Optimal, Unsat, Unknown}; Unknown
with a non-empty objective is renamed Satisfied so _compare treats it as
solved.

Usage:
    python aggregate.py            # default: test-orchestrator -> *.csv here
    python aggregate.py --data-root portfolios-final  # legacy layout
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

sys.path.insert(0, str(ROOT.parent.parent.parent))
from utils.borda import _compare, _parse_obj  # noqa: E402

DIR_RE = re.compile(r"^(?P<portfolio>.+)-(?P<year>\d{4})(?:-r(?P<rep>\d+))?$")
EXPECTED_REPS = 3
COMPLETE_STATUSES = {"Optimal", "Unsat"}
MAX_TIME_MS = 1_200_000

# Instance-level drop statuses: if any rep of any portfolio reports one of
# these on a (year, problem, name), drop the instance entirely from both
# outputs. WRAPPER_KILLED is an OOM kill — a property of the machine, not
# the solver, so the AI shouldn't see those instances at all.
DROP_INSTANCE_STATUSES = {"WRAPPER_KILLED"}


def load_problem_types(path: Path) -> dict[tuple[str, str], str]:
    if not path.exists():
        return {}
    with open(path) as f:
        return {(r["problem"], r["model"]): r["type"]
                for r in csv.DictReader(f)}


def translate_status(raw_status: str, objective: str) -> str:
    """Map results.csv 'optimal' field to the standard status set."""
    if raw_status == "Unknown" and objective not in ("", None):
        return "Satisfied"
    return raw_status


def parse_time(value: str) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def discover_runs(root: Path):
    """Yield (portfolio, year, rep, run_dir) for each results.csv."""
    for csv_path in sorted(root.rglob("results.csv")):
        run_dir = csv_path.parent
        m = DIR_RE.match(run_dir.name)
        if not m:
            print(f"WARN: skipping {run_dir} (unparseable name)", file=sys.stderr)
            continue
        yield m.group("portfolio"), m.group("year"), m.group("rep"), run_dir


def pick_median_rep(reps: list[dict], kind: str | None) -> dict:
    """Run a pairwise Borda tournament among reps and return the one whose
    total score is the median. With 3 reps that's the middle by sort order.

    Error reps are flagged as broken first (mirroring utils.borda.borda_scores):
    broken always loses to non-broken, broken-vs-broken ties at 0/0. Only
    after that filter do we call _compare for the ordinary comparison."""
    n = len(reps)
    scores = [0.0] * n
    broken = [r["status"] == "Error" for r in reps]
    for i in range(n):
        for j in range(i + 1, n):
            if broken[i] and broken[j]:
                sa, sb = 0.0, 0.0
            elif broken[i]:
                sa, sb = 0.0, 1.0
            elif broken[j]:
                sa, sb = 1.0, 0.0
            else:
                a, b = reps[i], reps[j]
                ta = parse_time(a["time_ms"]) or MAX_TIME_MS
                tb = parse_time(b["time_ms"]) or MAX_TIME_MS
                sa, sb = _compare(
                    a["status"], ta, _parse_obj(a["objective"]),
                    b["status"], tb, _parse_obj(b["objective"]),
                    kind,
                )
            scores[i] += sa
            scores[j] += sb
    order = sorted(range(n), key=lambda i: scores[i])
    return reps[order[n // 2]]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-dir",   type=Path, default=ROOT)
    args = ap.parse_args()

    problem_types = load_problem_types(TYPES_CSV)
    if not problem_types:
        print(f"WARNING: no problem types loaded from {TYPES_CSV}", file=sys.stderr)

    # {(portfolio, year, problem, name): {rep: row}}
    instance_reps: dict[tuple, dict[str, dict]] = defaultdict(dict)
    per_rep_rows: list[dict] = []

    for portfolio, year, rep, run_dir in discover_runs(args.data_root):
        if rep is None:
            print(f"WARN: {run_dir} has no rep suffix; skipping in test-orchestrator mode",
                  file=sys.stderr)
            continue
        with open(run_dir / "results.csv") as f:
            for r in csv.DictReader(f):
                status = translate_status(r["optimal"], r["objective"])
                key = (portfolio, year, r["problem"], r["name"])
                if rep in instance_reps[key]:
                    print(f"WARN: duplicate rep {rep} for {key}", file=sys.stderr)
                instance_reps[key][rep] = {
                    "schedule":  portfolio,
                    "model":     r["model"],
                    "time_ms":   r["time_ms"],
                    "objective": r["objective"],
                    "status":    status,
                }
                per_rep_rows.append({
                    "schedule":  portfolio,
                    "year":      year,
                    "rep":       rep,
                    "problem":   r["problem"],
                    "name":      r["name"],
                    "model":     r["model"],
                    "time_ms":   r["time_ms"],
                    "objective": r["objective"],
                    "status":    status,
                })

    missing = []
    for key, reps in instance_reps.items():
        if len(reps) != EXPECTED_REPS:
            missing.append((key, sorted(reps.keys())))
    if missing:
        print(f"\nERROR: {len(missing)} (portfolio, year, instance) groups don't have "
              f"{EXPECTED_REPS} reps:", file=sys.stderr)
        for key, reps_present in missing[:20]:
            print(f"  {key} -> reps {reps_present}", file=sys.stderr)
        if len(missing) > 20:
            print(f"  ... +{len(missing) - 20} more", file=sys.stderr)
        sys.exit(2)

    # Drop instances where any rep of any portfolio hit a DROP_INSTANCE_STATUSES
    # status (e.g. WRAPPER_KILLED = OOM). Drop the whole (year, problem, name)
    # across all portfolios so the AI sees a consistent instance set.
    tainted: set[tuple] = set()
    for (portfolio, year, problem, name), reps in instance_reps.items():
        if any(r["status"] in DROP_INSTANCE_STATUSES for r in reps.values()):
            tainted.add((year, problem, name))

    if tainted:
        n_before = len(instance_reps)
        instance_reps = {k: v for k, v in instance_reps.items()
                         if (k[1], k[2], k[3]) not in tainted}
        per_rep_rows = [r for r in per_rep_rows
                        if (r["year"], r["problem"], r["name"]) not in tainted]
        print(f"Dropped {len(tainted)} tainted (year, problem, name) instances "
              f"({n_before - len(instance_reps)} portfolio-rows removed)", file=sys.stderr)

    median_rows: list[dict] = []
    n_status_disagreements = 0
    for (portfolio, year, problem, name), reps in sorted(instance_reps.items()):
        rows = list(reps.values())
        kind = problem_types.get((problem, rows[0]["model"]))
        median = pick_median_rep(rows, kind)
        n_optimal = sum(1 for r in rows if r["status"] in COMPLETE_STATUSES)
        if len({r["status"] for r in rows}) > 1:
            n_status_disagreements += 1

        median_rows.append({
            "schedule":  median["schedule"],
            "year":      year,
            "problem":   problem,
            "name":      name,
            "model":     median["model"],
            "status":    median["status"],
            "objective": median["objective"],
            "time_ms":   median["time_ms"],
            "n_reps":    len(rows),
            "n_optimal": n_optimal,
        })

    combined_path = args.out_dir / "combined.csv"
    median_path = args.out_dir / "combined_median.csv"

    write_csv(combined_path,
              ["schedule", "year", "rep", "problem", "name", "model",
               "time_ms", "objective", "status"],
              per_rep_rows)
    write_csv(median_path,
              ["schedule", "year", "problem", "name", "model",
               "status", "objective", "time_ms", "n_reps", "n_optimal"],
              median_rows)

    print(f"Wrote {len(per_rep_rows):>5} rows to {combined_path.relative_to(ROOT)}")
    print(f"Wrote {len(median_rows):>5} rows to {median_path.relative_to(ROOT)}")
    portfolios = sorted({r["schedule"] for r in per_rep_rows})
    years = sorted({r["year"] for r in per_rep_rows})
    print(f"Portfolios:               {portfolios}")
    print(f"Years:                    {years}")
    print(f"Instances with rep status disagreement: {n_status_disagreements}")


if __name__ == "__main__":
    main()
