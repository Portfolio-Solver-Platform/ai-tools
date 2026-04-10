"""
Generates problem_types.csv from a directory of MiniZinc problem folders.

Each problem folder contains one or more .mzn files. The script parses
the solve statement in each model to determine the type (MIN/MAX/SAT).

Usage:
    python generate_problem_types.py [PROBLEMS_DIR]

Defaults to /Users/sofus/speciale/psp/problems
"""
import csv
import re
import sys
from pathlib import Path

DEFAULT_PROBLEMS_DIR = Path('/Users/sofus/speciale/psp/problems')
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


def main():
    problems_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PROBLEMS_DIR
    if not problems_dir.is_dir():
        print(f'Error: {problems_dir} is not a directory', file=sys.stderr)
        sys.exit(1)

    rows: list[tuple[str, str, str]] = []

    for problem_dir in sorted(problems_dir.iterdir()):
        if not problem_dir.is_dir():
            continue
        mzn_files = sorted(problem_dir.glob('*.mzn'))
        if not mzn_files:
            continue

        problem = problem_dir.name
        for mzn in mzn_files:
            kind = detect_type(mzn)
            if kind is None:
                continue
            model = mzn.stem
            rows.append((problem, model, kind))

    with open(OUTPUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['problem', 'model', 'type'])
        for row in rows:
            w.writerow(row)

    print(f'Wrote {len(rows)} entries to {OUTPUT_CSV}')

    unknowns = [r for r in rows if r[2] == '???']
    if unknowns:
        print(f'\nWARNING: {len(unknowns)} models with unknown type:')
        for prob, model, _ in unknowns:
            print(f'  {prob}/{model}')


if __name__ == '__main__':
    main()
