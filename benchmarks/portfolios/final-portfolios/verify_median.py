#!/usr/bin/env python3
"""
Randomly sample (portfolio, year, instance) groups, show all 3 rep rows
side by side with the median rep highlighted, and check the choice matches
the expected Borda ordering.

Also sweeps every group looking for picks that look "wrong" by the simple
rule: median rep's (status, objective, time) shouldn't be strictly better
than both others, nor strictly worse than both others.
"""
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent.parent))
from utils.borda import _compare, _parse_obj  # noqa: E402

COMBINED = ROOT / "combined.csv"
MEDIAN = ROOT / "combined_median.csv"
TYPES_CSV = ROOT.parent.parent / "open-category-benchmarks" / "problem_types.csv"
MAX_TIME_MS = 1_200_000


def load_problem_types():
    return {(r["problem"], r["model"]): r["type"]
            for r in csv.DictReader(open(TYPES_CSV))}


def load_per_rep():
    # {(schedule, year, problem, name): [rep_row, rep_row, rep_row]}
    by_key = defaultdict(list)
    for r in csv.DictReader(open(COMBINED)):
        by_key[(r["schedule"], r["year"], r["problem"], r["name"])].append(r)
    return by_key


def load_median():
    return {(r["schedule"], r["year"], r["problem"], r["name"]): r
            for r in csv.DictReader(open(MEDIAN))}


def score_pairwise(reps, kind):
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
                ta = int(a["time_ms"]) if a["time_ms"] else MAX_TIME_MS
                tb = int(b["time_ms"]) if b["time_ms"] else MAX_TIME_MS
                sa, sb = _compare(
                    a["status"], ta, _parse_obj(a["objective"]),
                    b["status"], tb, _parse_obj(b["objective"]),
                    kind,
                )
            scores[i] += sa
            scores[j] += sb
    return scores


def expected_median_score(scores):
    """The score value the picked rep must have to be 'a' median (handles
    ties: with scores 0/0/2 any of the two 0s is a valid median)."""
    return sorted(scores)[len(scores) // 2]


def fmt_rep(r, marker=""):
    return (f"  {marker}rep{r['rep']:>1}  status={r['status']:<10} "
            f"obj={(r['objective'] or '-'):>10}  time={r['time_ms']:>8}ms")


def picked_idx(reps, median_row):
    """Return the index of the rep that matches median_row's (status,
    objective, time). When multiple reps tie on those, returns the first;
    callers should compare by score not index."""
    for i, r in enumerate(reps):
        if (r["status"] == median_row["status"]
                and r["objective"] == median_row["objective"]
                and r["time_ms"] == median_row["time_ms"]):
            return i
    return None


def main():
    random.seed(42)
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 25

    types = load_problem_types()
    per_rep = load_per_rep()
    median = load_median()

    keys = list(per_rep.keys())
    sample = random.sample(keys, min(n_samples, len(keys)))

    print(f"=== Showing {len(sample)} random samples (seed=42) ===\n")
    for key in sample:
        sched, year, problem, name = key
        reps = sorted(per_rep[key], key=lambda r: r["rep"])
        kind = types.get((problem, reps[0]["model"]))
        med_row = median[key]

        scores = score_pairwise(reps, kind)
        med_score = expected_median_score(scores)
        got_idx = picked_idx(reps, med_row)
        got_score = scores[got_idx]

        ok = "OK " if got_score == med_score else "BAD"
        n_tied_at_median = sum(1 for s in scores if s == med_score)
        tie_note = f"  ({n_tied_at_median}-way tie)" if n_tied_at_median > 1 else ""
        print(f"{ok}  {sched}/{year}/{problem}/{name}  kind={kind}{tie_note}")
        for i, r in enumerate(reps):
            marker = "-> " if i == got_idx else "   "
            print(f"{marker}{fmt_rep(r)}  borda={scores[i]:.2f}")
        print()

    # Full sweep: count actual wrong picks (picked rep's score != median score)
    n_total = 0
    n_wrong = 0
    n_tied = 0
    for key, reps in per_rep.items():
        reps = sorted(reps, key=lambda r: r["rep"])
        kind = types.get((key[2], reps[0]["model"]))
        if key not in median:
            continue
        scores = score_pairwise(reps, kind)
        med_score = expected_median_score(scores)
        got_idx = picked_idx(reps, median[key])
        n_total += 1
        if scores[got_idx] != med_score:
            n_wrong += 1
            print(f"WRONG: {key}  scores={scores}  picked rep idx={got_idx}")
        if sum(1 for s in scores if s == med_score) > 1:
            n_tied += 1

    print(f"=== Full sweep: {n_wrong}/{n_total} wrong picks "
          f"({n_tied} had multiple reps tied at the median score) ===")


if __name__ == "__main__":
    main()
