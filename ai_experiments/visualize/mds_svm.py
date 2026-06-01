"""
3D MDS embedding of the cpsat8_k1 feature space (the same data svm.py
classifies). Points are colored by the winning portfolio (argmax of Borda).
"""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.shared_data import get_cpsat8_k1_data, prepare_labels


SEED = 42
OUT_PATH = Path(__file__).resolve().parent / "mds_cpsat8_k1.png"


def main():
    X, Y, meta = get_cpsat8_k1_data()
    y_labels, _ = prepare_labels(Y)

    Xs = StandardScaler().fit_transform(X)

    print(f"running MDS on {Xs.shape[0]} samples × {Xs.shape[1]} features...")
    mds = MDS(n_components=3, n_init=4, max_iter=300, n_jobs=-1,
              random_state=SEED, normalized_stress="auto")
    emb = mds.fit_transform(Xs)
    print(f"  done. stress = {mds.stress_:.3f}")

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    classes = np.unique(y_labels)
    palette = plt.get_cmap("tab10").colors
    for c, color in zip(classes, palette):
        m = y_labels == c
        ax.scatter(emb[m, 0], emb[m, 1], emb[m, 2], s=18, alpha=0.75,
                   color=color, label=f"portfolio {c}  (n={m.sum()})",
                   edgecolors="none", depthshade=True)
    ax.set_title("3D MDS embedding of cpsat8_k1 features\n"
                 f"colored by winning portfolio  (stress={mds.stress_:.2f})",
                 fontsize=13)
    ax.set_xlabel("MDS-1", fontsize=11)
    ax.set_ylabel("MDS-2", fontsize=11)
    ax.set_zlabel("MDS-3", fontsize=11)
    ax.view_init(elev=20, azim=-60)
    ax.legend(loc="upper left", frameon=True, fontsize=11)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"  saved {OUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
