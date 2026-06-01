#!/usr/bin/env bash
# Build the aggregated median dataset from a test-orchestrator-style data root.
#
# Usage:
#   ./build.sh                  # default data root: final-many-reps
#   ./build.sh some-other-dir   # any test-orchestrator-style tree
set -euo pipefail
cd "$(dirname "$0")"

DATA_ROOT="${1:-final-many-reps}"
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "data root not found: $DATA_ROOT" >&2
    exit 1
fi

echo "==> 1/5 aggregating from $DATA_ROOT"
python aggregate.py --data-root "$DATA_ROOT"

echo
echo "==> 2/5 building combined_sorted.csv"
python - <<'PY'
import csv
fields = ['schedule', 'year', 'rep', 'problem', 'name', 'model',
          'time_ms', 'objective', 'status']
rows = list(csv.DictReader(open('combined.csv')))
rows.sort(key=lambda r: (r['schedule'], r['year'], r['problem'], r['name'], r['rep']))
with open('combined_sorted.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(rows)
print(f'    wrote {len(rows)} rows to combined_sorted.csv')
PY

echo
echo "==> 3/5 verifying median picks"
python verify_median.py 0 | tail -1

echo
echo "==> 4/5 borda_relative (portfolios vs each other)"
python borda_relative.py

echo
echo "==> 5/5 borda_vs_open (portfolios vs open-category ladder, 2023-2025)"
python borda_vs_open.py
