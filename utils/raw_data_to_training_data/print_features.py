"""Print the 95-d mzn2feat feature vector for a single MiniZinc instance.

Usage:
    python print_features.py path/to/model.mzn
    python print_features.py path/to/model.mzn path/to/data.dzn
    python print_features.py path/to/model.mzn path/to/data.json
"""
import sys
from pathlib import Path

# Allow running this file directly without -m
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_features import generate_features


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(__doc__)
        sys.exit(1)

    mzn_path = sys.argv[1]
    data_path = sys.argv[2] if len(sys.argv) == 3 else None

    if not Path(mzn_path).exists():
        sys.exit(f"ERROR: mzn file not found: {mzn_path}")
    if data_path and not Path(data_path).exists():
        sys.exit(f"ERROR: data file not found: {data_path}")

    feat = generate_features(mzn_path, data_path, key="cli")
    if feat is None:
        sys.exit("ERROR: feature extraction failed (see messages above)")

    vec = feat.ravel()
    print(f"\nFeature vector ({len(vec)} dims) for {mzn_path}" + (f" with {data_path}" if data_path else ""))
    for i, v in enumerate(vec):
        print(f"  f{i:02d} = {v}")
    print(f"\nRaw comma-separated:\n{','.join(str(v) for v in vec)}")


if __name__ == "__main__":
    main()
