"""
Generates problem_types.csv from a directory of MiniZinc problem folders.

Each problem folder contains one or more .mzn files. The script parses
the solve statement in each model to determine the type (MIN/MAX/SAT).

Supports two layouts:
  - flat:        <problems_dir>/<problem>/*.mzn
  - year-nested: <problems_dir>/<year>/<problem>/*.mzn  (e.g. mzn-challenge)

Auto-detected: if every top-level subdir name is a 4-digit year, the
nested layout is used and results are deduped on (problem, model),
keeping the latest year's answer when types conflict.

Usage:
    python generate_problem_types.py [PROBLEMS_DIR]

Defaults to ~/speciale/ai/data/mzn-challenge
"""
import csv
import re
import sys
from pathlib import Path

DEFAULT_PROBLEMS_DIR = Path.home() / 'speciale' / 'ai' / 'data' / 'mzn-challenge'
OUTPUT_CSV = Path(__file__).parent.parent / 'problem_types.csv'


def strip_comments(text: str) -> str:
    """Remove MiniZinc line comments (%) and block comments (/* */)."""
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r'%.*', '', text)
    return text


def detect_type_from_text(text: str) -> str | None:
    """Return MIN, MAX, SAT, or None if text has no solve statement."""
    text = strip_comments(text)
    if not re.search(r'\bsolve\b', text):
        return None
    if re.search(r'\bminimize\b', text):
        return 'MIN'
    if re.search(r'\bmaximize\b', text):
        return 'MAX'
    if re.search(r'\bsatisfy\b', text):
        return 'SAT'
    return '???'


def detect_type(mzn_path: Path) -> str | None:
    """Return MIN, MAX, SAT, or None if file has no solve statement.

    If the file has no solve statement, follows include directives and
    checks .mzn.model files in the same directory.
    """
    text = mzn_path.read_text()
    result = detect_type_from_text(text)
    if result is not None:
        return result

    # Follow include directives
    for match in re.finditer(r'include\s+"([^"]+)"', text):
        included = mzn_path.parent / match.group(1)
        if included.exists():
            result = detect_type_from_text(included.read_text())
            if result is not None:
                return result

    return None


def problem_dirs(problems_dir: Path) -> list[Path]:
    """Return every problem folder under problems_dir, handling both layouts.

    Year-nested layout is auto-detected when every top-level subdir name
    is a 4-digit year. Problem dirs are returned in ascending-year order
    so later years win on dedupe.
    """
    subdirs = sorted(d for d in problems_dir.iterdir() if d.is_dir() and not d.name.startswith('.'))
    if subdirs and all(re.fullmatch(r'\d{4}', d.name) for d in subdirs):
        return [p for year in subdirs for p in sorted(year.iterdir()) if p.is_dir() and not p.name.startswith('.')]
    return subdirs


def main():
    problems_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PROBLEMS_DIR
    if not problems_dir.is_dir():
        print(f'Error: {problems_dir} is not a directory', file=sys.stderr)
        sys.exit(1)

    # (problem, model) -> (type, source_path) — later writes win, so we iterate
    # in ascending-year order to keep the most recent year's answer.
    seen: dict[tuple[str, str], tuple[str, Path]] = {}
    conflicts: list[str] = []

    for problem_dir in problem_dirs(problems_dir):
        mzn_files = sorted(problem_dir.glob('*.mzn'))
        if not mzn_files:
            continue

        problem = problem_dir.name
        for mzn in mzn_files:
            kind = detect_type(mzn)
            if kind is None:
                continue
            key = (problem, mzn.stem)
            prev = seen.get(key)
            if prev is not None and prev[0] != kind:
                conflicts.append(f'{problem}/{mzn.stem}: {prev[0]} ({prev[1]}) -> {kind} ({mzn})')
            seen[key] = (kind, mzn)

    rows = [(p, m, t) for (p, m), (t, _) in sorted(seen.items())]

    with open(OUTPUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['problem', 'model', 'type'])
        for row in rows:
            w.writerow(row)

    print(f'Wrote {len(rows)} entries to {OUTPUT_CSV}')

    if conflicts:
        print(f'\nWARNING: {len(conflicts)} (problem, model) pairs had conflicting types across years (latest kept):')
        for c in conflicts:
            print(f'  {c}')

    unknowns = [r for r in rows if r[2] == '???']
    if unknowns:
        print(f'\nWARNING: {len(unknowns)} models with unknown type:')
        for prob, model, _ in unknowns:
            print(f'  {prob}/{model}')


if __name__ == '__main__':
    main()
