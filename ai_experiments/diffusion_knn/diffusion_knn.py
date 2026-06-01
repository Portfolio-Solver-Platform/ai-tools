"""
Diffusion-maps embedding + KNN for cpsat8_k1 portfolio selection.

Hypothesis: KNN's Euclidean distance in raw feature space can miss the manifold
structure of problem-instance features. Re-embedding via diffusion maps puts
intrinsically-similar instances closer, so KNN should pick better neighbors.

Setup is transductive: the diffusion-map kernel is built over the union of
train/val/test feature vectors (no labels leak — eigenvectors depend only on
X). The KNN classifier itself only trains on training-set rows of the embedding.

Vanilla KNN runs side-by-side on the same 60/20/20 split for direct comparison.

Preprocessing note: features have huge dynamic range (some columns go up to 1e19),
so StandardScaler leaves enormous post-scale outliers that make the Gaussian kernel
collapse (exp(-huge) underflows → graph splits into isolated components → 12+
trivial eigenvalues of 1.0). We apply arcsinh (≈ identity near 0, ≈ log for large
|x|, smooth) then StandardScaler — tames outliers without flattening manifold
geometry the way per-feature quantile-mapping would. Applied to BOTH vanilla and
diffusion KNN so the comparison is fair.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.shared_data import get_cpsat8_k1_data, prepare_labels
from utils.cross_solver_eval import make_train_val_test_indices

N_TRIALS = 200
SEED = 42

optuna.logging.set_verbosity(optuna.logging.WARNING)


def diffuse(sq, eps_scale, alpha):
    """Coifman & Lafon diffusion operator on a precomputed sq-distance matrix.

    Returns (eigvals, psi) sorted descending, where psi columns are the right
    eigenvectors of the row-stochastic transition matrix P.
    """
    eps = eps_scale * np.median(sq[sq > 0])
    K = np.exp(-sq / eps)
    if alpha > 0:
        q = K.sum(axis=1) ** alpha
        K = K / np.outer(q, q)
    d_vec = K.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(d_vec)
    # Symmetric matrix similar to P; share eigenvalues.
    M_s = K * np.outer(d_inv_sqrt, d_inv_sqrt)
    w, v = np.linalg.eigh(M_s)
    order = np.argsort(w)[::-1]
    return w[order], v[:, order] * d_inv_sqrt[:, None]


def embedding(eigvals, psi, d, t):
    # Drop the trivial leading eigenvector (eigenvalue ≈ 1, constant).
    return psi[:, 1:d + 1] * (np.maximum(eigvals[1:d + 1], 0.0) ** t)


def main():
    X, Y, meta = get_cpsat8_k1_data()
    y, Y_borda = prepare_labels(Y)
    groups = meta["problem"]
    tr, va, te = make_train_val_test_indices(groups)
    trva = np.concatenate([tr, va])

    print(f"split sizes: train={len(tr)}  val={len(va)}  test={len(te)}")
    print(f"X: {X.shape}  classes: {np.bincount(y)}")

    Y_val = Y_borda[va]
    Y_test = Y_borda[te]
    oracle_val = float(np.sum(np.max(Y_val, axis=1)))
    oracle_test = float(np.sum(np.max(Y_test, axis=1)))
    cpsat_test = float(np.sum(Y_test[:, 0]))
    print(f"val  oracle: {oracle_val:.0f}")
    print(f"test oracle: {oracle_test:.0f}    cpsat8 baseline: {cpsat_test:.0f}")

    # arcsinh tames extreme outliers (some raw features go to 1e19) while
    # preserving local geometry, then standardize. Fit scaler on train rows only.
    X_asinh = np.arcsinh(X)
    scaler_tr = StandardScaler().fit(X_asinh[tr])
    X_std = scaler_tr.transform(X_asinh)

    print("\n----- vanilla KNN -----")
    def vanilla_obj(trial):
        knn = KNeighborsClassifier(
            n_neighbors=trial.suggest_int("n_neighbors", 1, 50),
            weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
            p=trial.suggest_categorical("p", [1, 2]),
        )
        knn.fit(X_std[tr], y[tr])
        pred = knn.predict(X_std[va])
        return float(np.sum(Y_val[np.arange(len(va)), pred]))

    s1 = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=SEED))
    s1.optimize(vanilla_obj, n_trials=N_TRIALS, show_progress_bar=False)
    print(f"  best val borda: {s1.best_value:.2f}    params: {s1.best_params}")

    scaler_trva = StandardScaler().fit(X_asinh[trva])
    X_trva_std = scaler_trva.transform(X_asinh)
    base = KNeighborsClassifier(**s1.best_params)
    base.fit(X_trva_std[trva], y[trva])
    base_pred = base.predict(X_trva_std[te])
    base_test = float(np.sum(Y_test[np.arange(len(te)), base_pred]))
    base_acc = float((base_pred == y[te]).mean())
    print(f"  test borda (refit train+val): {base_test:.2f}    acc={base_acc*100:.1f}%")

    print("\n----- diffusion-map KNN -----")
    sq = squareform(pdist(X_std, metric="sqeuclidean"))
    median_sq = float(np.median(sq[sq > 0]))
    print(f"  pdist done: n={sq.shape[0]}, median sq-dist = {median_sq:.3f}")

    # Cache eigendecompositions: same quantized (eps_scale, alpha) → reuse.
    cache = {}
    def get_emb(eps_scale, alpha, d, t):
        key = (round(float(np.log10(eps_scale)), 1), round(float(alpha), 1))
        if key not in cache:
            cache[key] = diffuse(sq, eps_scale, alpha)
        w, psi = cache[key]
        return embedding(w, psi, d, t)

    def dm_obj(trial):
        eps_scale = trial.suggest_float("eps_scale", 0.05, 20.0, log=True)
        alpha     = trial.suggest_float("alpha", 0.0, 1.0)
        d         = trial.suggest_int("d", 2, 40)
        t         = trial.suggest_int("t", 1, 5)
        n_neighbors = trial.suggest_int("n_neighbors", 1, 50)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        p       = trial.suggest_categorical("p", [1, 2])

        emb = get_emb(eps_scale, alpha, d, t)
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
        knn.fit(emb[tr], y[tr])
        pred = knn.predict(emb[va])
        return float(np.sum(Y_val[np.arange(len(va)), pred]))

    s2 = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=SEED))
    t0 = time.time()
    s2.optimize(dm_obj, n_trials=N_TRIALS, show_progress_bar=False)
    print(f"  HPO {time.time()-t0:.1f}s, {len(cache)} unique embeddings cached")
    print(f"  best val borda: {s2.best_value:.2f}    params: {s2.best_params}")

    bp = dict(s2.best_params)
    knn_kwargs = {k: bp.pop(k) for k in ("n_neighbors", "weights", "p")}
    emb_best = get_emb(bp["eps_scale"], bp["alpha"], bp["d"], bp["t"])
    w_best, _ = cache[(round(float(np.log10(bp["eps_scale"])), 1),
                       round(float(bp["alpha"]), 1))]
    print("  top-10 eigenvalues:  " + "  ".join(f"{wi:.3f}" for wi in w_best[:10]))

    dm = KNeighborsClassifier(**knn_kwargs)
    dm.fit(emb_best[trva], y[trva])
    dm_pred = dm.predict(emb_best[te])
    dm_test = float(np.sum(Y_test[np.arange(len(te)), dm_pred]))
    dm_acc = float((dm_pred == y[te]).mean())
    print(f"  test borda (refit train+val): {dm_test:.2f}    acc={dm_acc*100:.1f}%")

    print("\n========== Summary ==========")
    print(f"  oracle              : {oracle_test:.2f}")
    print(f"  cpsat8 baseline     : {cpsat_test:.2f}")
    print(f"  vanilla KNN         : {base_test:.2f}   ({base_test/oracle_test:.3f} of oracle,  acc {base_acc*100:.1f}%)")
    print(f"  diffusion-map KNN   : {dm_test:.2f}   ({dm_test/oracle_test:.3f} of oracle,  acc {dm_acc*100:.1f}%)")
    print(f"  Δ (DM − vanilla)    : {dm_test - base_test:+.2f}  borda  /  {(dm_acc-base_acc)*100:+.2f} pp acc")


if __name__ == "__main__":
    main()
