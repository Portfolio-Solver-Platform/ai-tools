#!/usr/bin/env python3
"""
Scan portfolios/combined.csv for data quality issues.

Checks per schedule+year group:
  - Time parsing issues (empty / non-numeric / negative)
  - Runaway times (>1,250,000ms)
  - Zero times
  - Unknown but exited early (<1,150,000ms)
  - Optimal in suspiciously short time (<100ms)
  - Optimal but no objective on MIN/MAX problems
  - Duplicate (problem, name) rows within a schedule+year
  - Rows flagged as wrong=True

Cross-schedule checks:
  - Same instance Optimal in two schedules but with different objectives

Run with --verbose to see every offending row; default output is a summary.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TYPES_CSV = ROOT.parent / "open-category-benchmarks" / "problem_types.csv"

TIMEOUT_MS = 1_200_000
EARLY_EXIT_THRESHOLD = 1_150_000
RUNAWAY_THRESHOLD     = 1_250_000
TINY_OPTIMAL_THRESHOLD = 100

SOLVED_STATUSES = {"Optimal", "Unsat", "Satisfied"}


def load_problem_types() -> dict[str, str]:
    types = {}
    if not TYPES_CSV.exists():
        return types
    with open(TYPES_CSV) as f:
        for r in csv.DictReader(f):
            types[r["model"]] = r["type"]
    return types


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir", type=Path,
                    help="data directory (e.g. all/ or eligible/)")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--filter", "-f", default=None,
                    help="only check schedules containing this substring")
    args = ap.parse_args()

    csv_path = args.data_dir / "combined.csv"

    problem_types = load_problem_types()
    if not problem_types:
        print(f"WARNING: no problem types loaded from {TYPES_CSV}", file=sys.stderr)

    with open(csv_path) as f:
        all_rows = list(csv.DictReader(f))

    # Group by schedule+year
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in all_rows:
        groups[(r["schedule"], r["year"])].append(r)

    grand_totals: dict[str, int] = defaultdict(int)

    for (schedule, year), rows in sorted(groups.items()):
        if args.filter and args.filter not in schedule:
            continue

        issues: dict[str, list[str]] = defaultdict(list)

        # Duplicates
        seen: dict[tuple, int] = defaultdict(int)
        for r in rows:
            seen[(r["problem"], r["name"])] += 1
        for k, v in seen.items():
            if v > 1:
                issues["duplicate_row"].append(f"{k[0]}/{k[1]} (x{v})")

        for r in rows:
            key = f"{r['problem']}/{r['name']}"
            kind = problem_types.get(r["model"])
            status = r["status"]
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

            if r.get("wrong", "").lower() == "true":
                issues["flagged_wrong"].append(f"{key} (from={r.get('last_result_from', '?')})")

        label = f"{schedule} ({year})"
        n = len(rows)
        cats = sorted((k, v) for k, v in issues.items() if v)
        if not cats:
            print(f"  {label}  ({n} rows)  ✔ clean")
        else:
            print(f"  {label}  ({n} rows)")
            for cat, items in cats:
                grand_totals[cat] += len(items)
                print(f"    • {cat}: {len(items)}")
                if args.verbose:
                    for it in items[:20]:
                        print(f"        - {it}")
                    if len(items) > 20:
                        print(f"        ... +{len(items) - 20} more")

    # Cross-schedule: conflicting optimal objectives
    print("\n═══ Cross-schedule checks ═══")
    by_instance: dict[tuple, list[tuple[str, str, str]]] = defaultdict(list)
    for r in all_rows:
        if r["status"] == "Optimal" and r["objective"]:
            by_instance[(r["problem"], r["name"], r["model"], r["year"])].append(
                (r["schedule"], r["objective"]))

    conflicts = 0
    for (problem, name, model, year), opts in sorted(by_instance.items()):
        objs = {o for _, o in opts}
        if len(objs) > 1:
            kind = problem_types.get(model, "?")
            detail = ", ".join(f"{s}={o}" for s, o in opts)
            if args.verbose:
                print(f"  conflicting_optimal: {problem}/{name} ({kind}, {year}): {detail}")
            conflicts += 1

    if conflicts:
        grand_totals["conflicting_optimal_objectives"] = conflicts
        if not args.verbose:
            print(f"  conflicting_optimal_objectives: {conflicts}")
    else:
        print("  ✔ no conflicting objectives")

    print("\n═══ GRAND TOTALS ═══")
    if not grand_totals:
        print("No issues found.")
    else:
        for k in sorted(grand_totals):
            print(f"  {k}: {grand_totals[k]}")


if __name__ == "__main__":
    main()
