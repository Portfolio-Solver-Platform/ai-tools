"""
Generate feature pickle for all instances in psp/problems.

Usage:
    python generate_psp_features.py [path_to_psp_problems]
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.raw_data_to_training_data.generate_features import generate_features_for_everything

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "psp_features"
DEFAULT_PROBLEMS = Path.home() / "speciale/psp/problems"


def main():
    problems = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PROBLEMS

    if not problems.is_dir():
        print(f"Problems directory not found: {problems}")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "psp_all_features.pkl"

    print(f"Generating features from {problems}")
    features = {}
    generate_features_for_everything(str(problems), features)

    with open(out, 'wb') as f:
        pickle.dump(features, f)
    print(f"Saved {len(features)} features to {out}")


if __name__ == "__main__":
    main()
