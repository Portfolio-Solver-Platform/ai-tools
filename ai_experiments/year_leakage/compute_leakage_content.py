from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
META_NPZ = Path("/home/sofus/speciale/ai/ai-tools/data/portfolios_cpsat8_k1_training_data.npz")
DATA_DIR = Path("/home/sofus/speciale/ai/data/mzn-challenge")
OUT_CSV = ROOT / "out" / "year_leakage_content.csv"

FAMILY_ALIASES = {
    "javarouting": "java-routing",
    "opt-cryptanalysis": "opt-cryptoanalysis",
    "rotating-workforce-scheduling": "rotating-workforce",
}


def norm_family(p):
    p = p.lower().replace("_", "-")
    return FAMILY_ALIASES.get(p, p)


def sha(p):
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    d = np.load(META_NPZ, allow_pickle=True)
    meta = d["meta"]

    rows = []
    for m in meta:
        y, p, mdl, n = int(m["year"]), str(m["problem"]), str(m["model"]), str(m["name"])
        folder = DATA_DIR / str(y) / p
        h_mzn = sha(folder / f"{mdl}.mzn")
        h_dzn = None
        for ext in (".dzn", ".json"):
            cand = folder / f"{n}{ext}"
            if cand.exists():
                h_dzn = sha(cand)
                break
        rows.append({
            "year":   y,
            "strict": (h_mzn, h_dzn) if h_mzn else None,
            "model":  h_mzn,
            "family": norm_family(p),
        })

    n_missing_mzn = sum(1 for r in rows if r["model"] is None)
    print(f"loaded {len(rows)} instances "
          f"({n_missing_mzn} missing .mzn files)\n")

    by_year_level = {
        "strict": defaultdict(set),
        "model":  defaultdict(set),
        "family": defaultdict(set),
    }
    for r in rows:
        for level in by_year_level:
            if r[level] is not None:
                by_year_level[level][r["year"]].add(r[level])

    years = sorted(by_year_level["strict"].keys())

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["year", "n",
                    "strict_n", "strict_pct",
                    "model_n",  "model_pct",
                    "family_n", "family_pct"])
        print(f"{'year':<5} {'n':>4}  "
              f"{'strict':>13} {'.mzn':>13} {'family':>13}")
        for y in years:
            year_rows = [r for r in rows if r["year"] == y]
            n_total = len(year_rows)
            row_out = [y, n_total]
            cells = []
            for level in ("strict", "model", "family"):
                others = set().union(*[by_year_level[level][yy]
                                       for yy in years if yy != y])
                n_hit = sum(1 for r in year_rows
                            if r[level] is not None and r[level] in others)
                pct = n_hit / n_total * 100
                row_out.extend([n_hit, f"{pct:.2f}"])
                cells.append(f"{n_hit:>3} ({pct:>5.1f}%)")
            w.writerow(row_out)
            print(f"{y:<5} {n_total:>4}  "
                  f"{cells[0]:>13} {cells[1]:>13} {cells[2]:>13}")

    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
