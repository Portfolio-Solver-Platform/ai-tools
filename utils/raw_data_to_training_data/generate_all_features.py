"""
Generate feature pickles for all years in mzn-challenge.

Usage:
    python generate_all_features.py [path_to_mzn_challenge]
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.raw_data_to_training_data.generate_features import generate_features_for_everything

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DEFAULT_BASE = Path.home() / "speciale/ai/data/mzn-challenge"


def main():
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BASE

    for year_dir in sorted(base.iterdir()):
        if not year_dir.is_dir() or year_dir.name.startswith('.'):
            continue

        out = DATA_DIR / f"mznc{year_dir.name}_features.pkl"
        if out.exists():
            print(f"Skipping {year_dir.name} (already exists: {out})")
            continue

        print(f"\n=== {year_dir.name} ===")
        features = {}
        generate_features_for_everything(str(year_dir), features)

        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'wb') as f:
            pickle.dump(features, f)
        print(f"Saved {len(features)} features to {out}")


if __name__ == "__main__":
    main()
