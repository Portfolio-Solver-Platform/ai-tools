from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
META_NPZ = Path("/home/sofus/speciale/ai/ai-tools/data/portfolios_cpsat8_k1_training_data.npz")
OUT_CSV = ROOT / "out" / "year_leakage.csv"

FAMILY_ALIASES = {
    "javarouting": "java-routing",
    "opt-cryptanalysis": "opt-cryptoanalysis",
    "rotating-workforce-scheduling": "rotating-workforce",
}


def norm_family(p: str) -> str:
    p = p.lower().replace("_", "-")
    return FAMILY_ALIASES.get(p, p)


DEFS = {
    "strict": lambda m: (str(m["problem"]), str(m["model"]), str(m["name"])),
    "model":  lambda m: (str(m["problem"]), str(m["model"])),
    "family": lambda m: (norm_family(str(m["problem"])),),
}


def main() -> None:
    d = np.load(META_NPZ, allow_pickle=True)
    meta = d["meta"]

    by_year: dict[str, dict[int, set]] = {k: defaultdict(set) for k in DEFS}
    for m in meta:
        y = int(m["year"])
        for k, f in DEFS.items():
            by_year[k][y].add(f(m))

    years = sorted(by_year["strict"].keys())

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "year", "n_instances",
            "strict_n", "strict_pct",
            "model_n",  "model_pct",
            "family_n", "family_pct",
        ])
        for y in years:
            row = [y, len(by_year["strict"][y])]
            for k in ("strict", "model", "family"):
                S = by_year[k][y]
                rest = set().union(*[by_year[k][yy] for yy in years if yy != y])
                overlap = len(S & rest)
                pct = overlap / max(len(S), 1) * 100.0
                row.extend([overlap, f"{pct:.2f}"])
            w.writerow(row)

    print(f"wrote {OUT_CSV}")
    with open(OUT_CSV) as f:
        for line in f:
            print(line, end="")


if __name__ == "__main__":
    main()
