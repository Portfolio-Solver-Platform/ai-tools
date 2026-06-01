"""
Nested CV for an MLP portfolio-selector. Mirrors svm.py for comparability.
Diverges from svm.py in three ways: soft-Borda targets blended with one-hot
argmax by an HPO knob, val-Borda early stopping (CE as tiebreaker), and
seed-ensemble inference to tame NN variance on this small dataset.
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
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.shared_data import (
    get_cpsat8_ek1_data, get_cpsat8_k1_data, get_cpsat8_k1_ek1_data, prepare_labels,
)
from utils.cross_solver_eval import year_kfold_folds


INNER_K   = 5
N_TRIALS  = 60
MAX_EPOCHS = 400
PATIENCE   = 30
EARLY_STOP_VAL_FRAC = 0.15
ENSEMBLE_K = 5
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)

DATASETS = [
    ('cpsat8_k1',     get_cpsat8_k1_data),
    # ('cpsat8_k1_ek1', get_cpsat8_k1_ek1_data),
    # ('cpsat8_ek1',    get_cpsat8_ek1_data),
]

optuna.logging.set_verbosity(optuna.logging.WARNING)


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
    """Mix one-hot argmax labels with row-normalized Borda distributions.
    hard_mix in [0, 1]: 0 = pure soft-Borda, 1 = pure one-hot argmax.
    Rows that sum to zero fall back to uniform on the soft side."""
    Y = Y_borda.astype(np.float32)
    row_sums = Y.sum(axis=1, keepdims=True)
    safe = np.where(row_sums > 0, row_sums, 1.0)
    soft = np.where(row_sums > 0, Y / safe, np.full_like(Y, 1.0 / Y.shape[1]))

    hard = np.zeros_like(Y)
    hard[np.arange(len(Y)), Y.argmax(axis=1)] = 1.0

    return (hard_mix * hard + (1.0 - hard_mix) * soft).astype(np.float32)


def train_mlp(X_tr, Y_borda_tr, num_classes, params, seed):
    _set_seed(seed)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X_tr))
    n_val = max(1, int(len(X_tr) * EARLY_STOP_VAL_FRAC))
    val_idx = order[:n_val]
    tr_idx  = order[n_val:]

    Y_targets   = _make_targets(Y_borda_tr, params["hard_mix"])
    Y_borda_val = Y_borda_tr[val_idx]
    val_rows    = np.arange(len(val_idx))

    Xt = _to_tensor(X_tr[tr_idx], torch.float32)
    yt = _to_tensor(Y_targets[tr_idx], torch.float32)
    Xv = _to_tensor(X_tr[val_idx], torch.float32)
    yv = _to_tensor(Y_targets[val_idx], torch.float32)

    model = MLP(
        in_dim=X_tr.shape[1],
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        num_classes=num_classes,
        dropout=params["dropout"],
    ).to(DEVICE)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    loss_fn = nn.CrossEntropyLoss()

    batch_size = min(params["batch_size"], len(tr_idx))
    n = len(tr_idx)

    # (val_borda, -val_ce); higher tuple is better.
    best_score = (-float("inf"), -float("inf"))
    best_state = None
    no_improve = 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch_size):
            sel = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[sel]), yt[sel])
            loss.backward()
            opt.step()

        model.train(False)
        with torch.no_grad():
            val_logits = model(Xv)
            val_ce = loss_fn(val_logits, yv).item()
            val_pred = val_logits.argmax(dim=1).cpu().numpy()
        val_borda = float(Y_borda_val[val_rows, val_pred].mean())
        score = (val_borda, -val_ce)
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_ensemble(X_tr, Y_borda_tr, num_classes, params):
    return [
        train_mlp(X_tr, Y_borda_tr, num_classes, params, seed=SEED + k)
        for k in range(ENSEMBLE_K)
    ]


def predict_ensemble(models, X):
    Xt = _to_tensor(X, torch.float32)
    probs_sum = None
    for m in models:
        m.train(False)
        with torch.no_grad():
            p = torch.softmax(m(Xt), dim=1)
            probs_sum = p if probs_sum is None else probs_sum + p
    return (probs_sum / len(models)).argmax(dim=1).cpu().numpy()


def cv_score(X, Y_borda, num_classes, params, splits):
    fold_means = []
    for tr, te in splits:
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr]).astype(np.float32)
        Xte = scaler.transform(X[te]).astype(np.float32)
        models = train_ensemble(Xtr, Y_borda[tr], num_classes, params)
        pred = predict_ensemble(models, Xte)
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
            "lr":           trial.suggest_float("lr", 1e-4, 5e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "batch_size":   trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "hard_mix":     trial.suggest_float("hard_mix", 0.0, 1.0),
        }
        return cv_score(X, Y_borda, num_classes, params, splits)

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return dict(study.best_params), study.best_value


def fit_final(X, Y_borda, num_classes, params):
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X).astype(np.float32)
    models = train_ensemble(Xs, Y_borda, num_classes, params)
    return scaler, models


def save_model(path, scaler, models, params, in_dim, num_classes):
    payload = {
        "scaler": scaler,
        "state_dicts": [
            {k: v.cpu() for k, v in m.state_dict().items()} for m in models
        ],
        "params": params,
        "in_dim": in_dim,
        "num_classes": num_classes,
        "architecture": "MLP",
        "ensemble_k": len(models),
    }
    joblib.dump(payload, path)


def load_model(path):
    payload = joblib.load(path)
    models = []
    for sd in payload["state_dicts"]:
        m = MLP(
            in_dim=payload["in_dim"],
            hidden_dim=payload["params"]["hidden_dim"],
            num_layers=payload["params"]["num_layers"],
            num_classes=payload["num_classes"],
            dropout=payload["params"]["dropout"],
        )
        m.load_state_dict(sd)
        m.train(False)
        models.append(m)
    return payload["scaler"], models


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

    for fold_label, train_idx, test_idx in folds:
        train_years = years[train_idx]
        best_params, best_inner_score = run_hpo(
            X[train_idx], Y_borda[train_idx], train_years,
            num_classes, n_splits=INNER_K, n_trials=N_TRIALS,
        )

        scaler, models = fit_final(
            X[train_idx], Y_borda[train_idx], num_classes, best_params,
        )
        pred = predict_ensemble(models, scaler.transform(X[test_idx]).astype(np.float32))

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
        }
        fold_records.append(record)

        print(f"    {fold_label}: borda={test_borda:>6.2f}  "
              f"oracle={oracle:>6.2f}  cpsat={cpsat_baseline:>6.2f}  "
              f"ratio={test_borda / oracle if oracle else float('nan'):.3f}  "
              f"acc={accuracy * 100:>5.1f}%  "
              f"(h={best_params['hidden_dim']}, L={best_params['num_layers']}, "
              f"d={best_params['dropout']:.2f}, lr={best_params['lr']:.1e}, "
              f"hard={best_params['hard_mix']:.2f})")

    sum_borda  = sum(r["test_borda"]     for r in fold_records)
    sum_oracle = sum(r["oracle"]         for r in fold_records)
    sum_cpsat  = sum(r["cpsat_baseline"] for r in fold_records)
    n_total    = sum(r["n_test"]         for r in fold_records)
    acc_weighted = sum(r["accuracy"] * r["n_test"] for r in fold_records) / n_total
    print(f"\n  totals: borda={sum_borda:.2f}  oracle={sum_oracle:.2f}  "
          f"cpsat={sum_cpsat:.2f}  oracle_ratio={sum_borda / sum_oracle:.3f}  "
          f"acc={acc_weighted * 100:.1f}%  ({n_total} test instances)")

    print("  fitting deliverable model (HPO on full data)...")
    final_params, final_score = run_hpo(
        X, Y_borda, years, num_classes,
        n_splits=INNER_K, n_trials=N_TRIALS,
    )
    scaler, models = fit_final(X, Y_borda, num_classes, final_params)

    models_dir = Path(__file__).resolve().parent / "models"
    models_dir.mkdir(exist_ok=True)
    out_path = models_dir / f"nn_model_{name}.joblib"
    save_model(out_path, scaler, models, final_params, X.shape[1], num_classes)
    print(f"  saved {out_path}  cv_score={final_score:.4f}  params={final_params}")

    return fold_records, final_params, final_score


def main():
    summary = []
    for name, getter in DATASETS:
        records, params, score = evaluate_dataset(name, getter)
        summary.append((name, records, params, score))

    print("\n========== Cross-dataset summary ==========")
    for name, records, params, score in summary:
        sb = sum(r["test_borda"]     for r in records)
        so = sum(r["oracle"]         for r in records)
        sc = sum(r["cpsat_baseline"] for r in records)
        print(f"  {name:18s}  borda={sb:8.2f}  oracle={so:8.2f}  "
              f"cpsat={sc:8.2f}  ratio={sb / so:.3f}  "
              f"deliverable: {params}")


if __name__ == "__main__":
    main()
