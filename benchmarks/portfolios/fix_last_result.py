"""
Retroactively fixes last_result_from for Unsat and Optimal entries
in results.csv by re-parsing the corresponding .out files.

For Unsat: uses the solver that claimed =====UNSATISFIABLE=====
For Optimal: uses the solver that claimed ==========

Usage:
    python fix_last_result.py all/
    python fix_last_result.py eligible/
"""
import argparse
import csv
import re
from pathlib import Path

STATUS_PATTERNS = {
    "Unsat": r"% NOTE: (.+?) got status =====UNSATISFIABLE=====",
    "Optimal": r"% NOTE: (.+?) got status ==========",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir", type=Path,
                    help="data directory (e.g. all/ or eligible/)")
    args = ap.parse_args()

    portfolios_dir = args.data_dir / "portfolios"
    fixed = 0
    total = 0

    for csv_path in sorted(portfolios_dir.rglob("results.csv")):
        rows = []
        changed = False

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                status = row["optimal"]
                if status in STATUS_PATTERNS:
                    total += 1
                    out_pattern = f"*-sep-*-sep-{row['name']}-sep-{row['schedule']}-sep-*-sep-*.out"
                    matches = list(csv_path.parent.glob(out_pattern))
                    if len(matches) == 1:
                        stdout = matches[0].read_text()
                        m = re.search(STATUS_PATTERNS[status], stdout)
                        if m:
                            old = row["last_result_from"]
                            new = m.group(1)
                            if old != new:
                                row["last_result_from"] = new
                                changed = True
                                fixed += 1
                    elif len(matches) == 0:
                        print(f"  WARNING: no .out file for {row['schedule']} {row['problem']}/{row['name']}")
                    else:
                        print(f"  WARNING: {len(matches)} .out files for {row['schedule']} {row['problem']}/{row['name']}")

                rows.append(row)

        if changed:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    print(f"Fixed {fixed}/{total} Unsat/Optimal entries")


if __name__ == "__main__":
    main()
