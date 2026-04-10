"""
For each solver, compute wrong-result and crash/error rates separately.

Table 1 — Wrong results: instances in wrong_results.csv (false UNSAT,
           wrong Optimal objective, etc.)
Table 2 — Errors/crashes: instances where status == "Error" in combined.csv.
"""
import csv
from collections import defaultdict
from pathlib import Path

CSV_PATH = Path(__file__).parent.parent / "open-category-benchmarks" / "combined.csv"
WRONG_CSV_PATH = Path(__file__).parent.parent / "open-category-benchmarks" / "wrong_results.csv"


def load_wrong_results(path: Path) -> set[tuple[str, str, str, str, str, str]]:
    wrong = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            wrong.add((row["solver"], row["cores"], row["year"],
                        row["problem"], row["model"], row["name"]))
    return wrong


def print_table(title: str, stats: dict, count_key: str) -> None:
    print(f"\n{'═' * 72}")
    print(f"  {title}")
    print(f"{'═' * 72}")
    print(f"{'Solver':<40} {'Cores':>5}  {count_key:>6} / {'Total':<6}  {'Rate':>7}")
    print("─" * 72)
    rows = sorted(
        stats.items(),
        key=lambda x: -(x[1][count_key] / x[1]["total"] if x[1]["total"] else 0),
    )
    for (solver, cores), s in rows:
        n = s[count_key]
        pct = n / s["total"] * 100 if s["total"] else 0.0
        print(f"{solver:<40} {cores:>5}  {n:>6} / {s['total']:<6}  {pct:>6.2f}%")


def main() -> None:
    wrong_results = load_wrong_results(WRONG_CSV_PATH)

    Key = tuple[str, str]
    stats: dict[Key, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "wrong": 0, "error": 0}
    )

    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            k: Key = (row["solver"], row["cores"])
            stats[k]["total"] += 1

            key = (row["solver"], row["cores"], row.get("year", ""),
                   row["problem"], row["model"], row["name"])
            if key in wrong_results:
                stats[k]["wrong"] += 1
            if row["status"] == "Error":
                stats[k]["error"] += 1

    print_table("Wrong results (from wrong_results.csv)", stats, "wrong")
    print_table("Errors / crashes (status=Error)", stats, "error")

    # Aggregate across cores
    avg: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "wrong": 0, "error": 0}
    )
    for (solver, _cores), s in stats.items():
        avg[solver]["total"] += s["total"]
        avg[solver]["wrong"] += s["wrong"]
        avg[solver]["error"] += s["error"]

    print_avg_table("Wrong results — averaged across cores", avg, "wrong")
    print_avg_table("Errors / crashes — averaged across cores", avg, "error")


def print_avg_table(title: str, stats: dict[str, dict[str, int]], count_key: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")
    print(f"{'Solver':<40}  {count_key:>6} / {'Total':<6}  {'Rate':>7}")
    print("─" * 60)
    rows = sorted(
        stats.items(),
        key=lambda x: -(x[1][count_key] / x[1]["total"] if x[1]["total"] else 0),
    )
    for solver, s in rows:
        n = s[count_key]
        pct = n / s["total"] * 100 if s["total"] else 0.0
        print(f"{solver:<40}  {n:>6} / {s['total']:<6}  {pct:>6.2f}%")


if __name__ == "__main__":
    main()
