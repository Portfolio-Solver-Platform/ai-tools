"""
Generate feature pickles for all years in mzn-challenge.

Usage:
    python generate_all_features.py [path_to_mzn_challenge]
"""
import pickle
import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.raw_data_to_training_data.generate_features import _generate_one, _build_jobs

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DEFAULT_BASE = Path.home() / "speciale/ai/data/mzn-challenge"


def main():
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BASE

    # Collect jobs from all years, tagging each with its year
    all_jobs = []  # (year, model, instance, key)
    years = []
    for year_dir in sorted(base.iterdir()):
        if not year_dir.is_dir() or year_dir.name.startswith('.'):
            continue

        out = DATA_DIR / f"mznc{year_dir.name}_features.pkl"
        if out.exists():
            print(f"Skipping {year_dir.name} (already exists: {out})")
            continue

        years.append(year_dir.name)
        jobs = _build_jobs(str(year_dir))
        for job in jobs:
            all_jobs.append((year_dir.name, job))

    if not all_jobs:
        print("Nothing to do")
        return

    print(f"Generating features for {len(all_jobs)} instances across years: {', '.join(years)}")

    n_workers = min(cpu_count(), len(all_jobs))

    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(_run_one, all_jobs),
            total=len(all_jobs),
        ))

    # Group results by year
    by_year = defaultdict(dict)
    for year, key, feature in results:
        if feature is not None:
            by_year[year][key] = feature

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for year in sorted(by_year):
        out = DATA_DIR / f"mznc{year}_features.pkl"
        with open(out, 'wb') as f:
            pickle.dump(by_year[year], f)
        print(f"Saved {len(by_year[year])} features to {out}")


def _run_one(args):
    year, job = args
    key, feature = _generate_one(job)
    return year, key, feature


if __name__ == "__main__":
    main()
