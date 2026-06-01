#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

DATA_ROOT="${1:-svc-2020-2022}"
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "data root not found: $DATA_ROOT" >&2
    echo "usage: $0 [data-root]   (default: svc-2020-2022)" >&2
    exit 1
fi

echo "==> 1/7 aggregating from $DATA_ROOT"
python3 aggregate.py --data-root "$DATA_ROOT"

echo
echo "==> 2/7 building combined_sorted.csv"
python3 - <<'PY'
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
echo "==> 3/7 verifying median picks"
python3 verify_median.py 0 | tail -1

echo
echo "==> 4/7 borda_relative (N-way Borda among portfolios in the data)"
python3 borda_relative.py

echo
echo "==> 5/7 borda_vs_cpsat (each AI vs cpsat8 baseline, per year)"
python3 borda_vs_cpsat.py

echo
echo "==> 6/8 borda_vs_cpsat_with_static (AI and static alt portfolios vs cpsat8)"
python3 borda_vs_cpsat_with_static.py

echo
echo "==> 7/8 borda_ai_vs_static (direct head-to-head: AI vs static alt portfolio)"
python3 borda_ai_vs_static.py

echo
echo "==> 8/8 decision-vs-overhead decomposition"
python3 borda_when_ai_picks_cpsat.py > /dev/null
python3 borda_when_ai_picks_alt.py   > /dev/null
python3 decision_vs_overhead.py

echo
echo "Done. Inputs: $DATA_ROOT/  Outputs: combined.csv, combined_median.csv,"
echo "                                     combined_sorted.csv,"
echo "                                     leaderboard_relative.csv,"
echo "                                     leaderboard_vs_cpsat.csv,"
echo "                                     leaderboard_vs_cpsat_with_static.csv,"
echo "                                     leaderboard_ai_vs_static.csv,"
echo "                                     leaderboard_when_cpsat.csv,"
echo "                                     leaderboard_when_alt.csv,"
echo "                                     decision_vs_overhead.csv,"
echo "                                     and matching .typ files"
