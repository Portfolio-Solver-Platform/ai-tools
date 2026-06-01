"""
Snapshot-ensemble MLP (Loshchilov-style SGDR). Single training run per HPO
config, CosineAnnealingWarmRestarts (T_0=CYCLE_EPOCHS, T_mult=1), snapshot
the state_dict at the end of each cycle, average softmaxes at inference.
HPO replaces `lr` with `lr_max`; otherwise mirrors nn.py.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.shared_data import (
    get_cpsat8_ek1_data, get_cpsat8_k1_data, get_cpsat8_k1_ek1_data, prepare_labels,
)
from utils.cross_solver_eval import year_kfold_folds


INNER_K       = 5
N_TRIALS      = 25
N_CYCLES      = 5
CYCLE_EPOCHS  = 60
SEED          = 0
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)

DATASETS = [
    ('cpsat8_k1', get_cpsat8_k1_data),
]

optuna.logging.set_verbosity(optuna.logging.WARNING)

_DIVERSITY_LOG = []


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden_dim
        layers.append(nn.Linear(d, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _to_tensor(arr, dtype):
    return torch.from_numpy(np.ascontiguousarray(arr)).to(dtype).to(DEVICE)


def _make_targets(Y_borda, hard_mix):
    Y = Y_borda.astype(np.float32)
    row_sums = Y.sum(axis=1, keepdims=True)
    safe = np.where(row_sums > 0, row_sums, 1.0)
    soft = np.where(row_sums > 0, Y / safe, np.full_like(Y, 1.0 / Y.shape[1]))

    hard = np.zeros_like(Y)
    hard[np.arange(len(Y)), Y.argmax(axis=1)] = 1.0

    return (hard_mix * hard + (1.0 - hard_mix) * soft).astype(np.float32)


def train_snapshot_ensemble(X_tr, Y_borda_tr, num_classes, params, seed,
                            n_cycles=N_CYCLES, cycle_epochs=CYCLE_EPOCHS):
    _set_seed(seed)

    Y_targets = _make_targets(Y_borda_tr, params["hard_mix"])
    Xt = _to_tensor(X_tr, torch.float32)
    yt = _to_tensor(Y_targets, torch.float32)

    model = MLP(
        in_dim=X_tr.shape[1],
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        num_classes=num_classes,
        dropout=params["dropout"],
    ).to(DEVICE)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=params["lr_max"],
        weight_decay=params["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=cycle_epochs, T_mult=1,
    )
    loss_fn = nn.CrossEntropyLoss()

    n = len(X_tr)
    batch_size = min(params["batch_size"], n)
    snapshots = []

    total_epochs = n_cycles * cycle_epochs
    for epoch in range(total_epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch_size):
            sel = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[sel]), yt[sel])
            loss.backward()
            opt.step()
        scheduler.step()

        # Snapshot at the bottom of the cosine: scheduler.step() has just
        # reset LR to lr_max, so the *previous* epoch's params are at the
        # cycle's minimum.
        if (epoch + 1) % cycle_epochs == 0:
            snapshots.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    return snapshots


def _build_model(params, in_dim, num_classes):
    return MLP(
        in_dim=in_dim,
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        num_classes=num_classes,
        dropout=params["dropout"],
    ).to(DEVICE)


def predict_snapshots(snapshots, params, in_dim, num_classes, X, return_probs=False):
    # Reuses a single model instance and swaps state_dicts to keep memory flat.
    Xt = _to_tensor(X, torch.float32)
    model = _build_model(params, in_dim, num_classes)
    probs_sum = None
    per_snap_probs = [] if return_probs else None
    for sd in snapshots:
        model.load_state_dict({k: v.to(DEVICE) for k, v in sd.items()})
        model.train(False)
        with torch.no_grad():
            p = torch.softmax(model(Xt), dim=1)
            if return_probs:
                per_snap_probs.append(p.cpu().numpy())
            probs_sum = p if probs_sum is None else probs_sum + p
    avg = (probs_sum / len(snapshots))
    pred = avg.argmax(dim=1).cpu().numpy()
    if return_probs:
        return pred, np.stack(per_snap_probs, axis=0)
    return pred


def cv_score(X, Y_borda, num_classes, params, splits):
    fold_means = []
    for tr, te in splits:
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr]).astype(np.float32)
        Xte = scaler.transform(X[te]).astype(np.float32)
        snapshots = train_snapshot_ensemble(
            Xtr, Y_borda[tr], num_classes, params, seed=SEED,
        )
        pred = predict_snapshots(snapshots, params, X.shape[1], num_classes, Xte)
        bordas = Y_borda[te][np.arange(len(te)), pred]
        fold_means.append(bordas.mean())
    return float(np.mean(fold_means))


def run_hpo(X, Y_borda, groups, num_classes, n_splits, n_trials):
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(X, groups=groups, y=groups))

    def objective(trial):
        params = {
            "hidden_dim":   trial.suggest_categorical("hidden_dim", [32, 64, 128, 256]),
            "num_layers":   trial.suggest_int("num_layers", 1, 4),
            "dropout":      trial.suggest_float("dropout", 0.0, 0.5),
            "lr_max":       trial.suggest_float("lr_max", 1e-3, 5e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "batch_size":   trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "hard_mix":     trial.suggest_float("hard_mix", 0.0, 1.0),
        }
        return cv_score(X, Y_borda, num_classes, params, splits)

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return dict(study.best_params), study.best_value


def _snapshot_diversity(per_snap_probs):
    """Mean pairwise cosine distance between snapshot prob vectors.
    0 = identical, 1 = orthogonal."""
    S, N, C = per_snap_probs.shape
    flat = per_snap_probs.reshape(S, N * C)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    flat_n = flat / np.where(norms > 0, norms, 1.0)
    sim = flat_n @ flat_n.T
    iu = np.triu_indices(S, k=1)
    return float(1.0 - sim[iu].mean())


def evaluate_dataset(name, getter):
    print(f"\n========== dataset: {name} ==========")
    X, Y, meta = getter()
    X = X.astype(np.float32)
    y_labels, Y_borda = prepare_labels(Y)
    years = meta["year"]
    num_classes = int(y_labels.max()) + 1

    folds = year_kfold_folds(years, n_splits=5)
    fold_records = []

    print(f"  outer folds: {len(folds)} (5-fold GroupKFold by year)")
    print(f"  in_dim={X.shape[1]}  num_classes={num_classes}  device={DEVICE}")
    print(f"  snapshots: N_CYCLES={N_CYCLES}  CYCLE_EPOCHS={CYCLE_EPOCHS}  "
          f"(total {N_CYCLES * CYCLE_EPOCHS} epochs/run)  N_TRIALS={N_TRIALS}")

    for fold_label, train_idx, test_idx in folds:
        train_years = years[train_idx]
        best_params, best_inner_score = run_hpo(
            X[train_idx], Y_borda[train_idx], train_years,
            num_classes, n_splits=INNER_K, n_trials=N_TRIALS,
        )

        scaler = StandardScaler().fit(X[train_idx])
        Xtr = scaler.transform(X[train_idx]).astype(np.float32)
        Xte = scaler.transform(X[test_idx]).astype(np.float32)
        snapshots = train_snapshot_ensemble(
            Xtr, Y_borda[train_idx], num_classes, best_params, seed=SEED,
        )
        pred, per_snap_probs = predict_snapshots(
            snapshots, best_params, X.shape[1], num_classes, Xte,
            return_probs=True,
        )
        diversity = _snapshot_diversity(per_snap_probs)
        _DIVERSITY_LOG.append((fold_label, diversity))

        Y_te = Y_borda[test_idx]
        test_borda     = Y_te[np.arange(len(test_idx)), pred].sum()
        oracle         = Y_te.max(axis=1).sum()
        cpsat_baseline = Y_te[:, 0].sum()
        accuracy       = (pred == y_labels[test_idx]).mean()

        record = {
            "fold_label":     fold_label,
            "n_test":         len(test_idx),
            "test_borda":     float(test_borda),
            "oracle":         float(oracle),
            "cpsat_baseline": float(cpsat_baseline),
            "accuracy":       float(accuracy),
            "best_params":    best_params,
            "inner_cv_score": float(best_inner_score),
            "diversity":      diversity,
        }
        fold_records.append(record)

        print(f"    {fold_label}: borda={test_borda:>6.2f}  "
              f"oracle={oracle:>6.2f}  cpsat={cpsat_baseline:>6.2f}  "
              f"ratio={test_borda / oracle if oracle else float('nan'):.3f}  "
              f"acc={accuracy * 100:>5.1f}%  div={diversity:.3f}  "
              f"(h={best_params['hidden_dim']}, L={best_params['num_layers']}, "
              f"d={best_params['dropout']:.2f}, lr_max={best_params['lr_max']:.1e}, "
              f"hard={best_params['hard_mix']:.2f})")

    sum_borda  = sum(r["test_borda"]     for r in fold_records)
    sum_oracle = sum(r["oracle"]         for r in fold_records)
    sum_cpsat  = sum(r["cpsat_baseline"] for r in fold_records)
    n_total    = sum(r["n_test"]         for r in fold_records)
    acc_weighted = sum(r["accuracy"] * r["n_test"] for r in fold_records) / n_total
    mean_div = float(np.mean([r["diversity"] for r in fold_records]))
    print(f"\n  totals: borda={sum_borda:.2f}  oracle={sum_oracle:.2f}  "
          f"cpsat={sum_cpsat:.2f}  oracle_ratio={sum_borda / sum_oracle:.3f}  "
          f"acc={acc_weighted * 100:.1f}%  ({n_total} test instances)")
    print(f"  mean snapshot diversity (1 - cosine sim of softmaxes): {mean_div:.3f}")

    return fold_records


def main():
    for name, getter in DATASETS:
        evaluate_dataset(name, getter)


if __name__ == "__main__":
    main()
