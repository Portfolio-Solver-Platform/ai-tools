"""
Nested CV for an FT-Transformer portfolio-selector. Same protocol as nn.py
(soft-Borda targets, val-Borda early stopping, seed ensembling); the
architecture differs: per-feature tokenizer + [CLS] token, pre-norm
transformer encoder, LayerNorm + Linear head on [CLS].
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


# Full-budget settings; assumes a CUDA GPU. CPU works but is slow.
INNER_K    = 5
N_TRIALS   = 15
MAX_EPOCHS = 200
PATIENCE   = 20
EARLY_STOP_VAL_FRAC = 0.15
ENSEMBLE_K     = 3
ENSEMBLE_K_HPO = 3
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASETS = [
    ('cpsat8_k1', get_cpsat8_k1_data),
]

optuna.logging.set_verbosity(optuna.logging.WARNING)


class FeatureTokenizer(nn.Module):
    """token_i = W_i * x_i + b_i, with a learnable [CLS] token prepended."""
    def __init__(self, n_features, d_model):
        super().__init__()
        # Initialize like nn.Linear (kaiming uniform); biases zero.
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias   = nn.Parameter(torch.zeros(n_features, d_model))
        self.cls    = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.cls,    a=5 ** 0.5)

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.weight + self.bias
        cls = self.cls.expand(x.size(0), -1, -1)
        return torch.cat([cls, tokens], dim=1)


class FTTransformer(nn.Module):
    def __init__(self, n_features, num_classes, d_model, n_blocks, n_heads,
                 ffn_hidden, attn_dropout, ffn_dropout):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_hidden,
            dropout=ffn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # TransformerEncoderLayer ties attn dropout to `dropout` by default;
        # decouple to match the original FT-T paper.
        encoder_layer.self_attn.dropout = attn_dropout
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        tokens = self.tokenizer(x)
        out = self.encoder(tokens)
        cls = out[:, 0]
        return self.head(cls)


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


def _resolve_heads(d_model, suggested):
    """Snap suggested n_heads down to the largest valid divisor of d_model.
    HPO allows [2, 4, 8] but only some divide d_model=96; fallback keeps
    Optuna's parameter space coherent."""
    for candidate in [suggested, 4, 2]:
        if d_model % candidate == 0:
            return candidate
    return 1


def _build_model(in_dim, num_classes, params):
    d_model = params["d_model"]
    n_heads = _resolve_heads(d_model, params["n_heads"])
    ffn_hidden = max(1, int(round(d_model * params["ffn_factor"])))
    return FTTransformer(
        n_features=in_dim,
        num_classes=num_classes,
        d_model=d_model,
        n_blocks=params["n_blocks"],
        n_heads=n_heads,
        ffn_hidden=ffn_hidden,
        attn_dropout=params["attn_dropout"],
        ffn_dropout=params["ffn_dropout"],
    ).to(DEVICE)


def train_ft(X_tr, Y_borda_tr, num_classes, params, seed):
    # Keep the checkpoint with highest val Borda (CE as tiebreaker).
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

    model = _build_model(X_tr.shape[1], num_classes, params)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    loss_fn = nn.CrossEntropyLoss()  # accepts soft targets (N, C)

    batch_size = min(params["batch_size"], len(tr_idx))
    n = len(tr_idx)

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


def train_ensemble(X_tr, Y_borda_tr, num_classes, params, k=None):
    if k is None:
        k = ENSEMBLE_K
    return [
        train_ft(X_tr, Y_borda_tr, num_classes, params, seed=SEED + i)
        for i in range(k)
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


def cv_score(X, Y_borda, num_classes, params, splits, ensemble_k=None):
    fold_means = []
    for tr, te in splits:
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr]).astype(np.float32)
        Xte = scaler.transform(X[te]).astype(np.float32)
        models = train_ensemble(Xtr, Y_borda[tr], num_classes, params, k=ensemble_k)
        pred = predict_ensemble(models, Xte)
        bordas = Y_borda[te][np.arange(len(te)), pred]
        fold_means.append(bordas.mean())
    return float(np.mean(fold_means))


def run_hpo(X, Y_borda, groups, num_classes, n_splits, n_trials, verbose=False):
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(X, groups=groups, y=groups))

    import time
    def objective(trial):
        params = {
            "d_model":      trial.suggest_categorical("d_model", [32, 64, 96]),
            "n_blocks":     trial.suggest_int("n_blocks", 1, 3),
            "n_heads":      trial.suggest_categorical("n_heads", [2, 4, 8]),
            "ffn_factor":   trial.suggest_float("ffn_factor", 1.0, 3.0),
            "attn_dropout": trial.suggest_float("attn_dropout", 0.0, 0.3),
            "ffn_dropout":  trial.suggest_float("ffn_dropout", 0.0, 0.3),
            "lr":           trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "batch_size":   trial.suggest_categorical("batch_size", [32, 64, 128]),
            "hard_mix":     trial.suggest_float("hard_mix", 0.0, 1.0),
        }
        t0 = time.time()
        score = cv_score(X, Y_borda, num_classes, params, splits,
                         ensemble_k=ENSEMBLE_K_HPO)
        if verbose:
            print(f"      trial {trial.number:3d}: score={score:.4f} "
                  f"d={params['d_model']} b={params['n_blocks']} "
                  f"h={params['n_heads']} bs={params['batch_size']} "
                  f"({time.time()-t0:.1f}s)", flush=True)
        return score

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return dict(study.best_params), study.best_value


def fit_final(X, Y_borda, num_classes, params):
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X).astype(np.float32)
    models = train_ensemble(Xs, Y_borda, num_classes, params)
    return scaler, models


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
    print(f"  budget: N_TRIALS={N_TRIALS}  MAX_EPOCHS={MAX_EPOCHS}  "
          f"PATIENCE={PATIENCE}  ENSEMBLE_K={ENSEMBLE_K} (HPO uses {ENSEMBLE_K_HPO})")

    for fold_label, train_idx, test_idx in folds:
        train_years = years[train_idx]
        print(f"    {fold_label}: starting HPO...", flush=True)
        best_params, best_inner_score = run_hpo(
            X[train_idx], Y_borda[train_idx], train_years,
            num_classes, n_splits=INNER_K, n_trials=N_TRIALS, verbose=True,
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

        resolved_heads = _resolve_heads(best_params["d_model"], best_params["n_heads"])
        print(f"    {fold_label}: borda={test_borda:>6.2f}  "
              f"oracle={oracle:>6.2f}  cpsat={cpsat_baseline:>6.2f}  "
              f"ratio={test_borda / oracle if oracle else float('nan'):.3f}  "
              f"acc={accuracy * 100:>5.1f}%  "
              f"(d={best_params['d_model']}, b={best_params['n_blocks']}, "
              f"h={resolved_heads}, ffn={best_params['ffn_factor']:.2f}, "
              f"lr={best_params['lr']:.1e}, hard={best_params['hard_mix']:.2f})")

    sum_borda  = sum(r["test_borda"]     for r in fold_records)
    sum_oracle = sum(r["oracle"]         for r in fold_records)
    sum_cpsat  = sum(r["cpsat_baseline"] for r in fold_records)
    n_total    = sum(r["n_test"]         for r in fold_records)
    acc_weighted = sum(r["accuracy"] * r["n_test"] for r in fold_records) / n_total
    print(f"\n  totals: borda={sum_borda:.2f}  oracle={sum_oracle:.2f}  "
          f"cpsat={sum_cpsat:.2f}  oracle_ratio={sum_borda / sum_oracle:.3f}  "
          f"acc={acc_weighted * 100:.1f}%  ({n_total} test instances)")

    return fold_records


def main():
    for name, getter in DATASETS:
        evaluate_dataset(name, getter)


if __name__ == "__main__":
    main()
