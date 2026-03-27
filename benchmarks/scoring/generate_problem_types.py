"""
Scans all .mzn files across the data year folders and produces problem_types.csv:
  model, type   (type is SAT, MIN, or MAX)
"""
import re
import csv
from pathlib import Path

DATA_PATH = Path('/Users/sofus/speciale/ai/data')
YEAR_FOLDERS = ['mznc2022_probs', 'mznc2023_probs', 'mznc2024_probs', 'mznc2025_probs']
OUT_PATH = Path(__file__).parent / 'problem_types.csv'


def get_solve_type(text: str) -> str | None:
    """Extract SAT/MIN/MAX from a .mzn file's solve item."""
    m = re.search(r'\bsolve\b', text)
    if m:
        after = text[m.start():]
        km = re.search(r'\b(minimize|maximize|satisfy)\b', after)
        if km:
            return {'minimize': 'MIN', 'maximize': 'MAX', 'satisfy': 'SAT'}[km.group(1)]
    return None


def main():
    results = {}
    conflicts = {}

    for year_folder in YEAR_FOLDERS:
        path = DATA_PATH / year_folder
        if not path.is_dir():
            continue
        for mzn in sorted(path.rglob('*.mzn')):
            text = mzn.read_text(encoding='utf-8', errors='replace')
            kind = get_solve_type(text)
            if kind is None:
                print(f'WARNING: could not determine type for {mzn}')
                continue
            model = mzn.stem
            if model in results and results[model] != kind:
                conflicts[model] = (results[model], kind)
            results[model] = kind

    if conflicts:
        print(f'WARNING: {len(conflicts)} models with conflicting types across years:')
        for m, (a, b) in conflicts.items():
            print(f'  {m}: {a} vs {b}')

    with open(OUT_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'type'])
        for model, kind in sorted(results.items()):
            writer.writerow([model, kind])

    print(f'Written {len(results)} entries to {OUT_PATH}')


if __name__ == '__main__':
    main()
