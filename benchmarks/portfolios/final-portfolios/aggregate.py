#!/usr/bin/env python3
"""
Aggregate the per-rep test-orchestrator results into two CSVs:

  combined.csv      one row per (portfolio, year, rep, instance)
  combined_avg.csv  one row per (portfolio, year, instance), reps merged

Rep aggregation in combined_avg.csv:
  status         majority status across reps (after Unknown+obj -> Satisfied
                 translation); empty string when all reps disagree
  status_error   True when no two reps agreed on status (the majority vote
                 has no winner); flagged because it shouldn't happen
  objective      mean of all numeric objectives across reps (regardless of
                 status); empty if no rep reported one
  time_ms_*      mean/min/max/stdev of valid times across reps
  n_reps         how many reps had a row for this instance
  n_optimal      how many reps reached Optimal or Unsat (proven complete)

Status translation: results.csv uses {Optimal, Unsat, Unknown}; an "Unknown"
row with a non-empty objective is renamed to "Satisfied" so downstream Borda
scoring (which uses {Optimal, Satisfied, Unsat, Unknown}) treats it as
solved.

Usage:
    python aggregate.py            # default: test-orchestrator -> *.csv here
    python aggregate.py --data-root portfolios-final  # legacy layout
"""
import argparse
import csv
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = ROOT / "test-orchestrator"
TYPES_CSV = ROOT.parent.parent / "open-category-benchmarks" / "problem_types.csv"

DIR_RE = re.compile(r"^(?P<portfolio>.+)-(?P<year>\d{4})(?:-r(?P<rep>\d+))?$")
EXPECTED_REPS = 3
COMPLETE_STATUSES = {"Optimal", "Unsat"}


def load_problem_types(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with open(path) as f:
        return {r["model"]: r["type"] for r in csv.DictReader(f)}


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


def majority_status(statuses: list[str]) -> tuple[str, bool]:
    """Return (status, error). With a majority, use it. Without a majority,
    if all statuses are on the feasibility axis {Optimal, Satisfied, Unknown}
    (no Unsat), fall back to Satisfied as the median. Anything else (e.g.
    Optimal vs Unsat conflict) is a real error."""
    counts = Counter(statuses)
    top_status, top_count = counts.most_common(1)[0]
    if top_count >= 2:
        return top_status, False
    if set(statuses) <= {"Optimal", "Satisfied", "Unknown"}:
        return "Satisfied", False
    return "", True


def mean_objective(rows: list[dict]) -> str:
    vals = []
    for r in rows:
        if r["objective"] in ("", None):
            continue
        try:
            vals.append(float(r["objective"]))
        except ValueError:
            pass
    if not vals:
        return ""
    m = sum(vals) / len(vals)
    return f"{m:.6g}"


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
                    "schedule": r["schedule"],
                    "model":    r["model"],
                    "time_ms":  r["time_ms"],
                    "objective": r["objective"],
                    "status":   status,
                }
                per_rep_rows.append({
                    "schedule": r["schedule"],
                    "year":     year,
                    "rep":      rep,
                    "problem":  r["problem"],
                    "name":     r["name"],
                    "model":    r["model"],
                    "time_ms":  r["time_ms"],
                    "objective": r["objective"],
                    "status":   status,
                })

    # Validate rep counts
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

    # Build aggregated rows
    avg_rows: list[dict] = []
    n_status_errors = 0
    for (portfolio, year, problem, name), reps in sorted(instance_reps.items()):
        rows = list(reps.values())
        statuses = [r["status"] for r in rows]
        status, status_error = majority_status(statuses)
        if status_error:
            n_status_errors += 1
            print(f"STATUS ERROR: {portfolio}/{year}/{problem}/{name}: "
                  f"reps={statuses}", file=sys.stderr)

        obj_mean = mean_objective(rows)

        times = [t for t in (parse_time(r["time_ms"]) for r in rows) if t is not None]
        if times:
            t_mean = round(statistics.mean(times))
            t_min = min(times)
            t_max = max(times)
            t_stdev = round(statistics.pstdev(times)) if len(times) > 1 else 0
        else:
            t_mean = t_min = t_max = t_stdev = ""

        n_optimal = sum(1 for s in statuses if s in COMPLETE_STATUSES)

        avg_rows.append({
            "schedule":      rows[0]["schedule"],
            "year":          year,
            "problem":       problem,
            "name":          name,
            "model":         rows[0]["model"],
            "status":        status,
            "status_error":  status_error,
            "objective":     obj_mean,
            "time_ms_mean":  t_mean,
            "time_ms_min":   t_min,
            "time_ms_max":   t_max,
            "time_ms_stdev": t_stdev,
            "n_reps":        len(rows),
            "n_optimal":     n_optimal,
        })

    combined_path = args.out_dir / "combined.csv"
    avg_path = args.out_dir / "combined_avg.csv"

    write_csv(combined_path,
              ["schedule", "year", "rep", "problem", "name", "model",
               "time_ms", "objective", "status"],
              per_rep_rows)
    write_csv(avg_path,
              ["schedule", "year", "problem", "name", "model",
               "status", "status_error", "objective",
               "time_ms_mean", "time_ms_min", "time_ms_max", "time_ms_stdev",
               "n_reps", "n_optimal"],
              avg_rows)

    print(f"Wrote {len(per_rep_rows):>5} rows to {combined_path.relative_to(ROOT)}")
    print(f"Wrote {len(avg_rows):>5} rows to {avg_path.relative_to(ROOT)}")
    portfolios = sorted({r["schedule"] for r in per_rep_rows})
    years = sorted({r["year"] for r in per_rep_rows})
    print(f"Portfolios:    {portfolios}")
    print(f"Years:         {years}")
    print(f"Status errors: {n_status_errors}"
          + ("  (no two reps agreed; should not happen!)" if n_status_errors else ""))


if __name__ == "__main__":
    main()
