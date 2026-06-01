"""Experiment registry for best_search."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler,
)
from sklearn.svm import SVC, SVR

from .harness import Experiment
from .preprocessing import Asinh, RankNormal, SignedLog1p

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    import catboost as cb
    _HAS_CB = True
except ImportError:
    _HAS_CB = False


def _pre(name: str):
    if name == "std":
        return [("scaler", StandardScaler())]
    if name == "robust":
        return [("scaler", RobustScaler(quantile_range=(5, 95)))]
    if name == "quantile":
        return [("scaler", QuantileTransformer(output_distribution="normal",
                                               n_quantiles=500, random_state=42))]
    if name == "power":
        return [("power", PowerTransformer(method="yeo-johnson"))]
    if name == "log_std":
        return [("log", SignedLog1p()), ("scaler", StandardScaler())]
    if name == "log_quantile":
        return [("log", SignedLog1p()),
                ("scaler", QuantileTransformer(output_distribution="normal",
                                               n_quantiles=500, random_state=42))]
    if name == "log_robust":
        return [("log", SignedLog1p()),
                ("scaler", RobustScaler(quantile_range=(5, 95)))]
    if name == "log_power":
        return [("log", SignedLog1p()),
                ("power", PowerTransformer(method="yeo-johnson"))]
    if name == "asinh_std":
        return [("asinh", Asinh()), ("scaler", StandardScaler())]
    if name == "asinh_quantile":
        return [("asinh", Asinh()),
                ("scaler", QuantileTransformer(output_distribution="normal",
                                               n_quantiles=500, random_state=42))]
    if name == "rank_normal":
        return [("rank", RankNormal())]
    if name == "rank_std":
        return [("rank", RankNormal()), ("scaler", StandardScaler())]
    raise ValueError(f"unknown preprocessing: {name}")


def svm_rbf(pre: str, n_trials: int = 100) -> Experiment:
    def build(p):
        steps = _pre(pre) + [("model", SVC(kernel="rbf", C=p["C"], gamma=p["gamma"],
                                           class_weight=p.get("class_weight")))]
        return Pipeline(steps)
    def suggest(trial):
        return {
            "C":            trial.suggest_float("C",     0.1, 100, log=True),
            "gamma":        trial.suggest_float("gamma", 1e-3, 1e1, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
    return Experiment(f"SVM-RBF/{pre}", build, suggest, n_trials=n_trials)


def svm_rbf_sample_weighted(pre: str, n_trials: int = 100) -> Experiment:
    """SVC weighted by |Y_borda[:,1] - Y_borda[:,0]| (regret margin)."""
    def build(p):
        steps = _pre(pre) + [("model", SVC(kernel="rbf", C=p["C"], gamma=p["gamma"]))]
        return Pipeline(steps)
    def suggest(trial):
        return {
            "C":     trial.suggest_float("C",     0.1, 100, log=True),
            "gamma": trial.suggest_float("gamma", 1e-3, 1e1, log=True),
            "wpow":  trial.suggest_float("wpow", 0.5, 2.5),
        }
    def fit_predict(pipe, X_tr, y_tr, Yb_tr, X_te):
        wpow = getattr(pipe, "_wpow", 1.0)
        w = np.abs(Yb_tr[:, 1] - Yb_tr[:, 0]) ** wpow + 1e-12
        pipe.fit(X_tr, y_tr, model__sample_weight=w)
        return pipe.predict(X_te)
    def build_wrapped(p):
        pipe = build(p)
        pipe._wpow = p["wpow"]
        return pipe
    return Experiment(f"SVM-RBF-MW/{pre}", build_wrapped, suggest,
                      n_trials=n_trials, fit_predict=fit_predict)


def svm_rbf_threshold(pre: str, n_trials: int = 100) -> Experiment:
    """SVC + probability threshold fallback to cpsat (column 0)."""
    def build(p):
        steps = _pre(pre) + [("model", SVC(kernel="rbf", C=p["C"], gamma=p["gamma"],
                                           probability=True, random_state=42))]
        pipe = Pipeline(steps)
        pipe._threshold = p["threshold"]
        return pipe
    def suggest(trial):
        return {
            "C":         trial.suggest_float("C",     0.1, 100, log=True),
            "gamma":     trial.suggest_float("gamma", 1e-3, 1e1, log=True),
            "threshold": trial.suggest_float("threshold", 0.5, 0.99),
        }
    def fit_predict(pipe, X_tr, y_tr, Yb_tr, X_te):
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)
        pred = np.argmax(proba, axis=1)
        pred[proba.max(axis=1) < pipe._threshold] = 0
        return pred
    return Experiment(f"SVM-RBF-T/{pre}", build, suggest,
                      n_trials=n_trials, fit_predict=fit_predict)


def mo_svr_borda(pre: str, n_trials: int = 80) -> Experiment:
    """Predict (cpsat_borda, k1_borda) directly; pick argmax."""
    def build(p):
        base = SVR(kernel="rbf", C=p["C"], gamma=p["gamma"], epsilon=p["epsilon"])
        steps = _pre(pre) + [("model", MultiOutputRegressor(base, n_jobs=1))]
        return Pipeline(steps)
    def suggest(trial):
        return {
            "C":       trial.suggest_float("C",       0.1, 100, log=True),
            "gamma":   trial.suggest_float("gamma",   1e-3, 1e1, log=True),
            "epsilon": trial.suggest_float("epsilon", 1e-3, 0.3, log=True),
        }
    def fit_predict(pipe, X_tr, y_tr, Yb_tr, X_te):
        pipe.fit(X_tr, Yb_tr)
        Y_pred = pipe.predict(X_te)
        return np.argmax(Y_pred, axis=1)
    return Experiment(f"MO-SVR/{pre}", build, suggest, n_trials=n_trials,
                      fit_predict=fit_predict)


def mo_xgb_borda(pre: str = "std", n_trials: int = 80) -> Experiment:
    if not _HAS_XGB:
        raise ImportError("xgboost not installed")
    def build(p):
        base = xgb.XGBRegressor(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"],
            min_child_weight=p["min_child_weight"],
            reg_lambda=p["reg_lambda"],
            tree_method="hist", n_jobs=1, random_state=42, verbosity=0,
        )
        steps = _pre(pre) + [("model", MultiOutputRegressor(base, n_jobs=1))]
        return Pipeline(steps)
    def suggest(trial):
        return {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 700),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
    def fit_predict(pipe, X_tr, y_tr, Yb_tr, X_te):
        pipe.fit(X_tr, Yb_tr)
        return np.argmax(pipe.predict(X_te), axis=1)
    return Experiment(f"MO-XGB/{pre}", build, suggest, n_trials=n_trials,
                      fit_predict=fit_predict)


def xgb_regret(pre: str = "std", n_trials: int = 100) -> Experiment:
    if not _HAS_XGB:
        raise ImportError("xgboost not installed")
    def build(p):
        steps = _pre(pre) + [("model", xgb.XGBClassifier(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"],
            min_child_weight=p["min_child_weight"],
            reg_lambda=p["reg_lambda"],
            tree_method="hist", n_jobs=1, random_state=42, verbosity=0,
        ))]
        pipe = Pipeline(steps)
        pipe._wpow = p["wpow"]
        return pipe
    def suggest(trial):
        return {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 700),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "wpow":             trial.suggest_float("wpow", 0.0, 2.5),
        }
    def fit_predict(pipe, X_tr, y_tr, Yb_tr, X_te):
        wpow = pipe._wpow
        if wpow == 0:
            pipe.fit(X_tr, y_tr)
        else:
            w = np.abs(Yb_tr[:, 1] - Yb_tr[:, 0]) ** wpow + 1e-12
            pipe.fit(X_tr, y_tr, model__sample_weight=w)
        return pipe.predict(X_te)
    return Experiment(f"XGB-MW/{pre}", build, suggest, n_trials=n_trials,
                      fit_predict=fit_predict)


def lgb_regret(pre: str = "std", n_trials: int = 100) -> Experiment:
    if not _HAS_LGB:
        raise ImportError("lightgbm not installed")
    def build(p):
        steps = _pre(pre) + [("model", lgb.LGBMClassifier(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            num_leaves=p["num_leaves"],
            learning_rate=p["learning_rate"],
            subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"],
            min_child_samples=p["min_child_samples"],
            reg_lambda=p["reg_lambda"],
            n_jobs=1, random_state=42, verbose=-1,
        ))]
        pipe = Pipeline(steps)
        pipe._wpow = p["wpow"]
        return pipe
    def suggest(trial):
        return {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "wpow":              trial.suggest_float("wpow", 0.0, 2.5),
        }
    def fit_predict(pipe, X_tr, y_tr, Yb_tr, X_te):
        wpow = pipe._wpow
        if wpow == 0:
            pipe.fit(X_tr, y_tr)
        else:
            w = np.abs(Yb_tr[:, 1] - Yb_tr[:, 0]) ** wpow + 1e-12
            pipe.fit(X_tr, y_tr, model__sample_weight=w)
        return pipe.predict(X_te)
    return Experiment(f"LGB-MW/{pre}", build, suggest, n_trials=n_trials,
                      fit_predict=fit_predict)


def et_regret(pre: str = "std", n_trials: int = 80) -> Experiment:
    def build(p):
        steps = _pre(pre) + [("model", ExtraTreesClassifier(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            min_samples_leaf=p["min_samples_leaf"],
            max_features=p["max_features"],
            class_weight=p.get("class_weight"),
            n_jobs=1, random_state=42,
        ))]
        pipe = Pipeline(steps)
        pipe._wpow = p["wpow"]
        return pipe
    def suggest(trial):
        return {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 700),
            "max_depth":        trial.suggest_int("max_depth", 5, 25),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":     trial.suggest_categorical("max_features",
                                                          ["sqrt", "log2", 0.5, 1.0]),
            "class_weight":     trial.suggest_categorical("class_weight",
                                                         [None, "balanced"]),
            "wpow":             trial.suggest_float("wpow", 0.0, 2.0),
        }
    def fit_predict(pipe, X_tr, y_tr, Yb_tr, X_te):
        wpow = pipe._wpow
        if wpow == 0:
            pipe.fit(X_tr, y_tr)
        else:
            w = np.abs(Yb_tr[:, 1] - Yb_tr[:, 0]) ** wpow + 1e-12
            pipe.fit(X_tr, y_tr, model__sample_weight=w)
        return pipe.predict(X_te)
    return Experiment(f"ET-MW/{pre}", build, suggest, n_trials=n_trials,
                      fit_predict=fit_predict)


def knn_exp(pre: str = "log_quantile", n_trials: int = 50) -> Experiment:
    def build(p):
        steps = _pre(pre) + [("model", KNeighborsClassifier(
            n_neighbors=p["k"], weights=p["weights"],
            metric=p["metric"], n_jobs=1,
        ))]
        return Pipeline(steps)
    def suggest(trial):
        return {
            "k":       trial.suggest_int("k", 1, 60),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric":  trial.suggest_categorical("metric", ["minkowski", "manhattan"]),
        }
    return Experiment(f"kNN/{pre}", build, suggest, n_trials=n_trials)


def diverse_bagged_svm_mw(pres: tuple[str, ...] = ("log_std", "quantile", "log_quantile"),
                          n_each: int = 5, n_trials: int = 80) -> Experiment:
    """Each preprocessing gets its own bag of n_each margin-weighted SVCs;
    one (C, gamma, wpow, max_samples) HPO shared across all of them.
    """
    def build(p):
        from sklearn.pipeline import Pipeline as P
        pipe = P([("ident", _Identity())])
        pipe._svm_params = {"C": p["C"], "gamma": p["gamma"]}
        pipe._wpow = p["wpow"]
        pipe._max_samples = p["max_samples"]
        pipe._pres = pres
        pipe._n_each = n_each
        return pipe

    def suggest(trial):
        return {
            "C":           trial.suggest_float("C",     0.1, 100, log=True),
            "gamma":       trial.suggest_float("gamma", 1e-3, 1e1, log=True),
            "wpow":        trial.suggest_float("wpow", 0.5, 2.5),
            "max_samples": trial.suggest_float("max_samples", 0.6, 1.0),
        }

    def fit_predict(pipe, X_tr, y_tr, Yb_tr, X_te):
        wpow = pipe._wpow
        max_samples = pipe._max_samples
        n_each = pipe._n_each
        params = pipe._svm_params
        pres_list = pipe._pres
        w_full = np.abs(Yb_tr[:, 1] - Yb_tr[:, 0]) ** wpow + 1e-12

        n = len(X_tr)
        sample_n = int(max_samples * n)
        idx_pos = np.where(y_tr == 1)[0]
        idx_neg = np.where(y_tr == 0)[0]
        n_pos = int(round(sample_n * len(idx_pos) / n))
        n_neg = sample_n - n_pos

        probs_sum = np.zeros((len(X_te), 2))
        seed_base = 0
        for pre_name in pres_list:
            steps = _pre(pre_name)
            pre_pipe = Pipeline(steps)
            pre_pipe.fit(X_tr)
            Xs_tr = pre_pipe.transform(X_tr)
            Xs_te = pre_pipe.transform(X_te)
            for k in range(n_each):
                seed = seed_base + k
                rng = np.random.default_rng(seed)
                sel_pos = rng.choice(idx_pos, size=n_pos, replace=True)
                sel_neg = rng.choice(idx_neg, size=n_neg, replace=True)
                sel = np.concatenate([sel_pos, sel_neg])
                base = SVC(kernel="rbf", probability=True,
                           random_state=seed, **params)
                base.fit(Xs_tr[sel], y_tr[sel], sample_weight=w_full[sel])
                probs_sum += base.predict_proba(Xs_te)
            seed_base += n_each
        return np.argmax(probs_sum, axis=1)

    tag = "+".join(pres)
    return Experiment(f"DivBag/{tag}", build, suggest, n_trials=n_trials,
                      fit_predict=fit_predict)


def bagged_svm_mw(pre: str = "quantile", n_estimators: int = 10,
                  n_trials: int = 60) -> Experiment:
    """Average predict_proba across K bootstrap-sampled, margin-weighted SVCs.

    Bootstrap is stratified on the binary label.
    """
    def build(p):
        steps = _pre(pre)
        pipe = Pipeline(steps + [("ident", _Identity())])
        pipe._svm_params = {"C": p["C"], "gamma": p["gamma"]}
        pipe._wpow = p["wpow"]
        pipe._n_estimators = n_estimators
        pipe._max_samples = p["max_samples"]
        pipe._pre_name = pre
        pipe._models = []
        return pipe

    def suggest(trial):
        return {
            "C":           trial.suggest_float("C",     0.1, 100, log=True),
            "gamma":       trial.suggest_float("gamma", 1e-3, 1e1, log=True),
            "wpow":        trial.suggest_float("wpow", 0.5, 2.5),
            "max_samples": trial.suggest_float("max_samples", 0.6, 1.0),
        }

    def fit_predict(pipe, X_tr, y_tr, Yb_tr, X_te):
        pre_pipe = Pipeline(pipe.steps[:-1])  # everything except _Identity
        pre_pipe.fit(X_tr)
        Xs_tr = pre_pipe.transform(X_tr)
        Xs_te = pre_pipe.transform(X_te)

        wpow = pipe._wpow
        n_est = pipe._n_estimators
        max_samples = pipe._max_samples
        params = pipe._svm_params
        w_full = np.abs(Yb_tr[:, 1] - Yb_tr[:, 0]) ** wpow + 1e-12

        n = len(X_tr)
        sample_n = int(max_samples * n)
        idx_pos = np.where(y_tr == 1)[0]
        idx_neg = np.where(y_tr == 0)[0]
        n_pos = int(round(sample_n * len(idx_pos) / n))
        n_neg = sample_n - n_pos

        probs_sum = np.zeros((len(X_te), 2))
        for seed in range(n_est):
            rng = np.random.default_rng(seed)
            sel_pos = rng.choice(idx_pos, size=n_pos, replace=True)
            sel_neg = rng.choice(idx_neg, size=n_neg, replace=True)
            sel = np.concatenate([sel_pos, sel_neg])
            base = SVC(kernel="rbf", probability=True, random_state=seed,
                       **params)
            base.fit(Xs_tr[sel], y_tr[sel], sample_weight=w_full[sel])
            probs_sum += base.predict_proba(Xs_te)
        return np.argmax(probs_sum, axis=1)

    return Experiment(f"BagSVM-MW/{pre}", build, suggest, n_trials=n_trials,
                      fit_predict=fit_predict)


class _Identity:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


def catboost_exp(pre: str = "std", n_trials: int = 60) -> Experiment:
    if not _HAS_CB:
        raise ImportError("catboost not installed")
    def build(p):
        steps = _pre(pre) + [("model", cb.CatBoostClassifier(
            iterations=p["iterations"],
            depth=p["depth"],
            learning_rate=p["learning_rate"],
            l2_leaf_reg=p["l2_leaf_reg"],
            random_strength=p["random_strength"],
            bagging_temperature=p["bagging_temperature"],
            auto_class_weights=p.get("class_weights", None),
            random_seed=42, verbose=0, thread_count=1,
        ))]
        pipe = Pipeline(steps)
        pipe._wpow = p["wpow"]
        return pipe
    def suggest(trial):
        return {
            "iterations":          trial.suggest_int("iterations", 100, 800),
            "depth":               trial.suggest_int("depth", 3, 10),
            "learning_rate":       trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 1e-2, 30, log=True),
            "random_strength":     trial.suggest_float("random_strength", 1e-3, 10, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "class_weights":       trial.suggest_categorical("class_weights",
                                                             [None, "Balanced"]),
            "wpow":                trial.suggest_float("wpow", 0.0, 2.0),
        }
    def fit_predict(pipe, X_tr, y_tr, Yb_tr, X_te):
        wpow = pipe._wpow
        if wpow == 0:
            pipe.fit(X_tr, y_tr)
        else:
            w = np.abs(Yb_tr[:, 1] - Yb_tr[:, 0]) ** wpow + 1e-12
            pipe.fit(X_tr, y_tr, model__sample_weight=w)
        return pipe.predict(X_te)
    return Experiment(f"CatBoost/{pre}", build, suggest, n_trials=n_trials,
                      fit_predict=fit_predict)


def tabpfn_exp(pre: str = "std", n_trials: int = 6) -> Experiment:
    from tabpfn import TabPFNClassifier
    def build(p):
        steps = _pre(pre) + [("model", TabPFNClassifier(
            n_estimators=p["n_estimators"],
            softmax_temperature=p["softmax_temperature"],
            ignore_pretraining_limits=True,
            random_state=42, device="cpu",
        ))]
        pipe = Pipeline(steps)
        pipe._wpow = p["wpow"]
        return pipe
    def suggest(trial):
        return {
            "n_estimators":        trial.suggest_int("n_estimators", 4, 16),
            "softmax_temperature": trial.suggest_float("softmax_temperature", 0.5, 1.5),
            "wpow":                trial.suggest_float("wpow", 0.0, 2.0),
        }
    def fit_predict(pipe, X_tr, y_tr, Yb_tr, X_te):
        wpow = pipe._wpow
        if wpow == 0:
            pipe.fit(X_tr, y_tr)
        else:
            w = np.abs(Yb_tr[:, 1] - Yb_tr[:, 0]) ** wpow + 1e-12
            try:
                pipe.fit(X_tr, y_tr, model__sample_weight=w)
            except Exception:
                pipe.fit(X_tr, y_tr)
        return pipe.predict(X_te)
    return Experiment(f"TabPFN/{pre}", build, suggest, n_trials=n_trials,
                      fit_predict=fit_predict)


def random_subspace_svm_mw(pre: str = "log_std", n_estimators: int = 30,
                           n_trials: int = 60) -> Experiment:
    """BagSVM-MW where each base sees a random subset of features AND a bootstrap."""
    def build(p):
        return p

    def suggest(trial):
        return {
            "C":            trial.suggest_float("C",     0.5, 50, log=True),
            "gamma":        trial.suggest_float("gamma", 1e-3, 1, log=True),
            "wpow":         trial.suggest_float("wpow", 0.5, 2.5),
            "max_samples":  trial.suggest_float("max_samples", 0.6, 1.0),
            "max_features": trial.suggest_float("max_features", 0.4, 1.0),
        }

    def fit_predict(params, X_tr, y_tr, Yb_tr, X_te):
        pre_pipe = Pipeline(_pre(pre)); pre_pipe.fit(X_tr)
        Xs_tr = pre_pipe.transform(X_tr); Xs_te = pre_pipe.transform(X_te)
        w = np.abs(Yb_tr[:, 1] - Yb_tr[:, 0]) ** params["wpow"] + 1e-12

        n = len(X_tr); n_feat = Xs_tr.shape[1]
        sample_n = int(params["max_samples"] * n)
        feat_n = max(2, int(params["max_features"] * n_feat))
        idx_pos = np.where(y_tr == 1)[0]; idx_neg = np.where(y_tr == 0)[0]
        n_pos = int(round(sample_n * len(idx_pos)/n))
        n_neg = sample_n - n_pos

        probs = np.zeros((len(X_te), 2))
        for seed in range(n_estimators):
            rng = np.random.default_rng(seed)
            sel = np.concatenate([rng.choice(idx_pos, size=n_pos, replace=True),
                                  rng.choice(idx_neg, size=n_neg, replace=True)])
            feat_sel = rng.choice(n_feat, size=feat_n, replace=False)
            base = SVC(kernel="rbf", probability=True, random_state=seed,
                       C=params["C"], gamma=params["gamma"])
            base.fit(Xs_tr[sel][:, feat_sel], y_tr[sel], sample_weight=w[sel])
            probs += base.predict_proba(Xs_te[:, feat_sel])
        return np.argmax(probs, axis=1)

    return Experiment(f"RandSubspace-MW/{pre}-n{n_estimators}", build, suggest,
                      n_trials=n_trials, fit_predict=fit_predict)


def recency_bag_svm_mw(pre: str = "log_std", n_estimators: int = 15,
                       n_trials: int = 60) -> Experiment:
    """BagSVM-MW with sample weights that decay older training instances.

    weight_i = (margin_i)^wpow * exp(alpha * (year_i - year_min))
    """
    def build(p):
        return p

    def suggest(trial):
        return {
            "C":           trial.suggest_float("C",     0.5, 50, log=True),
            "gamma":       trial.suggest_float("gamma", 1e-3, 1, log=True),
            "wpow":        trial.suggest_float("wpow", 0.5, 2.5),
            "max_samples": trial.suggest_float("max_samples", 0.6, 1.0),
            "alpha":       trial.suggest_float("alpha", 0.0, 0.5),
        }

    def fit_predict(params, X_tr, y_tr, Yb_tr, X_te, years_tr=None):
        # Harness wrapper injects training years via attribute on fit_predict.
        years = getattr(fit_predict, "_RECENCY_YEARS_TR", None)
        if years is None:
            raise RuntimeError("recency_bag_svm_mw needs year context")

        pre_pipe = Pipeline(_pre(pre)); pre_pipe.fit(X_tr)
        Xs_tr = pre_pipe.transform(X_tr); Xs_te = pre_pipe.transform(X_te)
        margin = np.abs(Yb_tr[:, 1] - Yb_tr[:, 0])
        w_margin = (margin ** params["wpow"]) + 1e-12
        rel_year = (years - years.min()).astype(np.float64)
        w_rec = np.exp(params["alpha"] * rel_year)
        w_full = w_margin * w_rec

        n = len(X_tr); sample_n = int(params["max_samples"] * n)
        idx_pos = np.where(y_tr == 1)[0]; idx_neg = np.where(y_tr == 0)[0]
        n_pos = int(round(sample_n * len(idx_pos)/n))
        n_neg = sample_n - n_pos
        probs = np.zeros((len(X_te), 2))
        for seed in range(n_estimators):
            rng = np.random.default_rng(seed)
            sel = np.concatenate([rng.choice(idx_pos, size=n_pos, replace=True),
                                  rng.choice(idx_neg, size=n_neg, replace=True)])
            base = SVC(kernel="rbf", probability=True, random_state=seed,
                       C=params["C"], gamma=params["gamma"])
            base.fit(Xs_tr[sel], y_tr[sel], sample_weight=w_full[sel])
            probs += base.predict_proba(Xs_te)
        return np.argmax(probs, axis=1)

    return Experiment(f"RecencyBag-MW/{pre}", build, suggest,
                      n_trials=n_trials, fit_predict=fit_predict)


def fe_bag_svm_mw(pre: str = "log_std", n_estimators: int = 10,
                  n_trials: int = 60, k_top: int = 20) -> Experiment:
    """BagSVM-MW with polynomial(degree=2) interactions on top-k Spearman features."""
    from sklearn.preprocessing import PolynomialFeatures
    from scipy.stats import spearmanr

    def build(p):
        return {**p, "n_estimators": n_estimators, "k_top": k_top}

    def suggest(trial):
        return {
            "C":           trial.suggest_float("C",     0.5, 50, log=True),
            "gamma":       trial.suggest_float("gamma", 1e-3, 1, log=True),
            "wpow":        trial.suggest_float("wpow", 0.5, 2.5),
            "max_samples": trial.suggest_float("max_samples", 0.6, 1.0),
            "interactions_only": trial.suggest_categorical("interactions_only", [True, False]),
        }

    def fit_predict(params, X_tr, y_tr, Yb_tr, X_te):
        pre_pipe = Pipeline(_pre(pre)); pre_pipe.fit(X_tr)
        Xs_tr = pre_pipe.transform(X_tr); Xs_te = pre_pipe.transform(X_te)

        margin = Yb_tr[:, 1] - Yb_tr[:, 0]
        corrs = np.zeros(Xs_tr.shape[1])
        for i in range(Xs_tr.shape[1]):
            try:
                r, _ = spearmanr(Xs_tr[:, i], margin)
                corrs[i] = abs(r) if not np.isnan(r) else 0
            except Exception:
                corrs[i] = 0
        top_idx = np.argsort(corrs)[-params["k_top"] if "k_top" in params else -k_top:]

        poly = PolynomialFeatures(
            degree=2,
            interaction_only=params["interactions_only"],
            include_bias=False,
        )
        Xtr_top = poly.fit_transform(Xs_tr[:, top_idx])
        Xte_top = poly.transform(Xs_te[:, top_idx])
        Xtr_full = np.hstack([Xs_tr, Xtr_top])
        Xte_full = np.hstack([Xs_te, Xte_top])

        w = np.abs(Yb_tr[:, 1] - Yb_tr[:, 0]) ** params["wpow"] + 1e-12
        n = len(X_tr); sample_n = int(params["max_samples"] * n)
        idx_pos = np.where(y_tr == 1)[0]; idx_neg = np.where(y_tr == 0)[0]
        n_pos = int(round(sample_n * len(idx_pos)/n))
        n_neg = sample_n - n_pos
        probs = np.zeros((len(X_te), 2))
        for seed in range(params["n_estimators"]):
            rng = np.random.default_rng(seed)
            sel = np.concatenate([rng.choice(idx_pos, size=n_pos, replace=True),
                                  rng.choice(idx_neg, size=n_neg, replace=True)])
            base = SVC(kernel="rbf", probability=True, random_state=seed,
                       C=params["C"], gamma=params["gamma"])
            base.fit(Xtr_full[sel], y_tr[sel], sample_weight=w[sel])
            probs += base.predict_proba(Xte_full)
        return np.argmax(probs, axis=1)

    return Experiment(f"FE-BagSVM/{pre}", build, suggest, n_trials=n_trials,
                      fit_predict=fit_predict)


def mixture_experts_exp(pre: str = "log_std", n_trials: int = 60,
                       n_estimators: int = 10, n_clusters: int = 3) -> Experiment:
    """K-means clustering on features; one bagged SVM-MW expert per cluster.

    Falls back to a global bag when a cluster is too small.
    """
    from sklearn.cluster import KMeans

    def build(p):
        return {**p, "n_estimators": n_estimators, "n_clusters": n_clusters}

    def suggest(trial):
        return {
            "C":           trial.suggest_float("C",     0.5, 50, log=True),
            "gamma":       trial.suggest_float("gamma", 1e-3, 1, log=True),
            "wpow":        trial.suggest_float("wpow", 0.5, 2.5),
            "max_samples": trial.suggest_float("max_samples", 0.6, 1.0),
        }

    def _fit_bag(Xs_tr, y_tr, w_full, n_est, max_samples, params):
        n = len(Xs_tr); sample_n = int(max_samples * n)
        idx_pos = np.where(y_tr == 1)[0]; idx_neg = np.where(y_tr == 0)[0]
        if len(idx_pos) == 0 or len(idx_neg) == 0:
            return None
        n_pos = max(1, int(round(sample_n * len(idx_pos)/n)))
        n_neg = max(1, sample_n - n_pos)
        models = []
        for seed in range(n_est):
            rng = np.random.default_rng(seed)
            sel = np.concatenate([rng.choice(idx_pos, size=n_pos, replace=True),
                                  rng.choice(idx_neg, size=n_neg, replace=True)])
            base = SVC(kernel="rbf", probability=True, random_state=seed,
                       C=params["C"], gamma=params["gamma"])
            base.fit(Xs_tr[sel], y_tr[sel], sample_weight=w_full[sel])
            models.append(base)
        return models

    def fit_predict(params, X_tr, y_tr, Yb_tr, X_te):
        pre_pipe = Pipeline(_pre(pre)); pre_pipe.fit(X_tr)
        Xs_tr = pre_pipe.transform(X_tr); Xs_te = pre_pipe.transform(X_te)
        w = np.abs(Yb_tr[:, 1] - Yb_tr[:, 0]) ** params["wpow"] + 1e-12

        n_clust = params["n_clusters"]
        km = KMeans(n_clusters=n_clust, random_state=42, n_init=5)
        labels_tr = km.fit_predict(Xs_tr)
        labels_te = km.predict(Xs_te)

        global_models = _fit_bag(Xs_tr, y_tr, w, params["n_estimators"],
                                  params["max_samples"], params)

        cluster_models = {}
        for c in range(n_clust):
            m = labels_tr == c
            if m.sum() < 20:
                cluster_models[c] = None
                continue
            cluster_models[c] = _fit_bag(Xs_tr[m], y_tr[m], w[m],
                                          params["n_estimators"],
                                          params["max_samples"], params)

        probs = np.zeros((len(X_te), 2))
        for c in range(n_clust):
            m = labels_te == c
            if not m.any():
                continue
            experts = cluster_models[c] or global_models
            p = np.zeros((m.sum(), 2))
            for base in experts:
                p += base.predict_proba(Xs_te[m])
            probs[m] = p / len(experts)
        return np.argmax(probs, axis=1)

    return Experiment(f"MoE/{pre}-k{n_clusters}", build, suggest,
                      n_trials=n_trials, fit_predict=fit_predict)


def hp_diverse_bag_svm_mw(pre: str = "log_std", n_estimators: int = 50,
                           n_trials: int = 60) -> Experiment:
    """Bagged SVM-MW where each base draws (C, gamma) from a tunable log-uniform range."""
    def build(p):
        return p

    def suggest(trial):
        return {
            "C_lo":        trial.suggest_float("C_lo",    0.3, 5.0, log=True),
            "C_hi":        trial.suggest_float("C_hi",    5.0, 50.0, log=True),
            "gamma_lo":    trial.suggest_float("gamma_lo", 1e-3, 0.02, log=True),
            "gamma_hi":    trial.suggest_float("gamma_hi", 0.02, 0.5, log=True),
            "wpow":        trial.suggest_float("wpow", 0.5, 2.5),
            "max_samples": trial.suggest_float("max_samples", 0.6, 1.0),
        }

    def fit_predict(params, X_tr, y_tr, Yb_tr, X_te):
        pre_pipe = Pipeline(_pre(pre)); pre_pipe.fit(X_tr)
        Xs_tr = pre_pipe.transform(X_tr); Xs_te = pre_pipe.transform(X_te)
        w = np.abs(Yb_tr[:, 1] - Yb_tr[:, 0]) ** params["wpow"] + 1e-12

        n = len(X_tr); sample_n = int(params["max_samples"] * n)
        idx_pos = np.where(y_tr == 1)[0]; idx_neg = np.where(y_tr == 0)[0]
        n_pos = int(round(sample_n * len(idx_pos)/n))
        n_neg = sample_n - n_pos

        rng_master = np.random.default_rng(0)
        Cs = np.exp(rng_master.uniform(np.log(params["C_lo"]),
                                       np.log(params["C_hi"]), n_estimators))
        Gs = np.exp(rng_master.uniform(np.log(params["gamma_lo"]),
                                       np.log(params["gamma_hi"]), n_estimators))

        probs = np.zeros((len(X_te), 2))
        for seed in range(n_estimators):
            rng = np.random.default_rng(seed)
            sel = np.concatenate([rng.choice(idx_pos, size=n_pos, replace=True),
                                  rng.choice(idx_neg, size=n_neg, replace=True)])
            base = SVC(kernel="rbf", probability=True, random_state=seed,
                       C=float(Cs[seed]), gamma=float(Gs[seed]))
            base.fit(Xs_tr[sel], y_tr[sel], sample_weight=w[sel])
            probs += base.predict_proba(Xs_te)
        return np.argmax(probs, axis=1)

    return Experiment(f"HPDivBag-MW/{pre}-n{n_estimators}", build, suggest,
                      n_trials=n_trials, fit_predict=fit_predict)


def outlier_combo_exp(pre: str = "log_std", n_trials: int = 60,
                     n_estimators: int = 10) -> Experiment:
    """Bagged SVM-MW gated by an IsolationForest trained on cpsat-wins.

    Decision rule:
        P(k1) >= theta1  AND  (P(k1) >= theta2  OR  outlier_score >= s_thr)
    """
    from sklearn.ensemble import IsolationForest

    def build(p):
        return p

    def suggest(trial):
        return {
            "C":            trial.suggest_float("C",     0.5, 30, log=True),
            "gamma":        trial.suggest_float("gamma", 1e-3, 1, log=True),
            "wpow":         trial.suggest_float("wpow", 0.5, 2.5),
            "max_samples":  trial.suggest_float("max_samples", 0.6, 1.0),
            "iso_contam":   trial.suggest_float("iso_contam", 0.05, 0.4),
            "theta1":       trial.suggest_float("theta1", 0.05, 0.50),
            "theta2":       trial.suggest_float("theta2", 0.40, 0.95),
            "use_outlier":  trial.suggest_categorical("use_outlier", [True, False]),
        }

    def fit_predict(params, X_tr, y_tr, Yb_tr, X_te):
        pre_pipe = Pipeline(_pre(pre)); pre_pipe.fit(X_tr)
        Xs_tr = pre_pipe.transform(X_tr); Xs_te = pre_pipe.transform(X_te)

        w = np.abs(Yb_tr[:, 1] - Yb_tr[:, 0]) ** params["wpow"] + 1e-12
        n = len(X_tr); sample_n = int(params["max_samples"] * n)
        idx_pos = np.where(y_tr == 1)[0]; idx_neg = np.where(y_tr == 0)[0]
        n_pos = int(round(sample_n * len(idx_pos)/n))
        n_neg = sample_n - n_pos
        probs = np.zeros((len(X_te), 2))
        for seed in range(n_estimators):
            rng = np.random.default_rng(seed)
            sel = np.concatenate([rng.choice(idx_pos, size=n_pos, replace=True),
                                  rng.choice(idx_neg, size=n_neg, replace=True)])
            base = SVC(kernel="rbf", probability=True, random_state=seed,
                       C=params["C"], gamma=params["gamma"])
            base.fit(Xs_tr[sel], y_tr[sel], sample_weight=w[sel])
            probs += base.predict_proba(Xs_te)
        probs /= n_estimators
        p_k1 = probs[:, 1]

        cpsat_mask = (y_tr == 0)
        iso = IsolationForest(
            contamination=params["iso_contam"], random_state=42, n_jobs=1,
            n_estimators=100, max_samples="auto",
        )
        iso.fit(Xs_tr[cpsat_mask])
        scores_te = iso.score_samples(Xs_te)
        scores_tr_in = iso.score_samples(Xs_tr[cpsat_mask])
        from numpy import searchsorted
        sorted_tr = np.sort(scores_tr_in)
        pct = searchsorted(sorted_tr, scores_te) / len(sorted_tr)
        # smaller pct = more outlier-like
        outlier_pct = 1 - pct

        theta1, theta2 = params["theta1"], params["theta2"]
        if params["use_outlier"]:
            pred = ((p_k1 >= theta2) | ((p_k1 >= theta1) & (outlier_pct >= 0.5))).astype(int)
        else:
            pred = (p_k1 >= 0.5).astype(int)
        return pred

    return Experiment(f"OutlierCombo/{pre}", build, suggest, n_trials=n_trials,
                      fit_predict=fit_predict)


def pairwise_rank_exp(pre: str = "log_std", n_trials: int = 60,
                     n_estimators: int = 10) -> Experiment:
    """Bagged SVR on the Borda margin (Y_borda[k1] - Y_borda[cpsat]); threshold at tau."""
    from sklearn.svm import SVR

    def build(p):
        return p

    def suggest(trial):
        return {
            "C":           trial.suggest_float("C",     0.1, 50, log=True),
            "gamma":       trial.suggest_float("gamma", 1e-3, 1, log=True),
            "epsilon":     trial.suggest_float("epsilon", 1e-3, 0.3, log=True),
            "max_samples": trial.suggest_float("max_samples", 0.6, 1.0),
            "margin_pow":  trial.suggest_float("margin_pow", 0.5, 2.5),
            "tau":         trial.suggest_float("tau", -0.1, 0.1),
        }

    def fit_predict(params, X_tr, y_tr, Yb_tr, X_te):
        pre_pipe = Pipeline(_pre(pre)); pre_pipe.fit(X_tr)
        Xs_tr = pre_pipe.transform(X_tr); Xs_te = pre_pipe.transform(X_te)
        margin_tr = Yb_tr[:, 1] - Yb_tr[:, 0]  # positive => k1 wins
        w = np.abs(margin_tr) ** params["margin_pow"] + 1e-12

        n = len(X_tr); sample_n = int(params["max_samples"] * n)
        all_idx = np.arange(n)
        pred_margins = np.zeros(len(X_te))
        for seed in range(n_estimators):
            rng = np.random.default_rng(seed)
            sel = rng.choice(all_idx, size=sample_n, replace=True)
            svr = SVR(kernel="rbf", C=params["C"], gamma=params["gamma"],
                      epsilon=params["epsilon"])
            svr.fit(Xs_tr[sel], margin_tr[sel], sample_weight=w[sel])
            pred_margins += svr.predict(Xs_te)
        pred_margins /= n_estimators
        return (pred_margins > params["tau"]).astype(int)

    return Experiment(f"PairwiseRank/{pre}", build, suggest, n_trials=n_trials,
                      fit_predict=fit_predict)


def regret_mlp_exp(pre: str = "log_std", n_trials: int = 50,
                   n_estimators: int = 5) -> Experiment:
    """MLP minimising expected Borda regret: loss = mean_i sum_c q_i(c) * (max(Y_i) - Y_i[c])."""
    import torch
    import torch.nn as nn
    DEVICE = torch.device("cpu")

    class MLP(nn.Module):
        def __init__(self, d_in, hidden, n_layers, dropout, n_out):
            super().__init__()
            layers = []; d = d_in
            for _ in range(n_layers):
                layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
                d = hidden
            layers.append(nn.Linear(d, n_out))
            self.net = nn.Sequential(*layers)
        def forward(self, x): return self.net(x)

    def _train_one(X_tr, Yb_tr, params, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(X_tr))
        n_val = max(1, int(0.15 * len(X_tr)))
        v, t = order[:n_val], order[n_val:]
        Xt = torch.from_numpy(X_tr[t]).float().to(DEVICE)
        Yt = torch.from_numpy(Yb_tr[t]).float().to(DEVICE)
        Xv = torch.from_numpy(X_tr[v]).float().to(DEVICE)
        Yv_borda = Yb_tr[v]

        model = MLP(X_tr.shape[1], params["hidden"], params["n_layers"],
                    params["dropout"], Yb_tr.shape[1]).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=params["lr"],
                                weight_decay=params["wd"])
        Ymax_t = Yt.max(dim=1, keepdim=True).values
        regret_t = Ymax_t - Yt  # (n, 2)
        best_borda = -1.0; best_state = None; patience = 0
        bs = min(params["batch_size"], len(t))
        for epoch in range(params["max_epochs"]):
            model.train()
            perm = torch.randperm(len(t))
            for i in range(0, len(t), bs):
                sel = perm[i:i+bs]
                logits = model(Xt[sel])
                q = torch.softmax(logits / params["temp"], dim=1)
                loss = (q * regret_t[sel]).sum(dim=1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            model.train(False)
            with torch.no_grad():
                pred = model(Xv).argmax(dim=1).cpu().numpy()
            v_borda = float(Yv_borda[np.arange(len(v)), pred].mean())
            if v_borda > best_borda:
                best_borda = v_borda; patience = 0
                best_state = {k: v_.detach().clone() for k, v_ in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 20: break
        if best_state: model.load_state_dict(best_state)
        return model

    def build(p):
        return p

    def suggest(trial):
        return {
            "hidden":     trial.suggest_categorical("hidden", [32, 64, 128, 256]),
            "n_layers":   trial.suggest_int("n_layers", 1, 4),
            "dropout":    trial.suggest_float("dropout", 0.0, 0.5),
            "lr":         trial.suggest_float("lr", 1e-4, 5e-2, log=True),
            "wd":         trial.suggest_float("wd", 1e-6, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "max_epochs": trial.suggest_int("max_epochs", 80, 250),
            "temp":       trial.suggest_float("temp", 0.5, 3.0),
        }

    def fit_predict(params, X_tr, y_tr, Yb_tr, X_te):
        pre_pipe = Pipeline(_pre(pre)); pre_pipe.fit(X_tr)
        Xs_tr = pre_pipe.transform(X_tr).astype(np.float32)
        Xs_te = pre_pipe.transform(X_te).astype(np.float32)

        probs_sum = np.zeros((len(X_te), Yb_tr.shape[1]), dtype=np.float64)
        for k in range(n_estimators):
            m = _train_one(Xs_tr, Yb_tr.astype(np.float32), params, seed=k)
            m.train(False)
            with torch.no_grad():
                Xt = torch.from_numpy(Xs_te).float().to(DEVICE)
                q = torch.softmax(m(Xt) / params["temp"], dim=1).cpu().numpy()
            probs_sum += q
        return np.argmax(probs_sum, axis=1)

    return Experiment(f"RegretMLP/{pre}", build, suggest, n_trials=n_trials,
                      fit_predict=fit_predict)


def gpc_exp(pre: str = "log_std", n_trials: int = 12) -> Experiment:
    """GPC with tunable RBF length-scale. n_restarts_optimizer=0 for speed."""
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    def build(p):
        kernel = ConstantKernel(p["c0"]) * RBF(length_scale=p["length_scale"])
        steps = _pre(pre) + [("model", GaussianProcessClassifier(
            kernel=kernel, n_restarts_optimizer=0, random_state=42,
            max_iter_predict=100,
        ))]
        return Pipeline(steps)
    def suggest(trial):
        return {
            "c0":           trial.suggest_float("c0", 0.1, 10, log=True),
            "length_scale": trial.suggest_float("length_scale", 0.1, 50, log=True),
        }
    return Experiment(f"GPC/{pre}", build, suggest, n_trials=n_trials)


def calibrated_bag_svm_mw(pre: str = "log_std", n_estimators: int = 10,
                          n_trials: int = 60) -> Experiment:
    """Bagged SVM-MW with isotonic recalibration fit on inner-OOF predict_proba."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import GroupKFold

    def build(p):
        steps = _pre(pre)
        pipe = Pipeline(steps + [("ident", _Identity())])
        pipe._svm_params = {"C": p["C"], "gamma": p["gamma"]}
        pipe._wpow = p["wpow"]
        pipe._n_estimators = n_estimators
        pipe._max_samples = p["max_samples"]
        pipe._pre_name = pre
        return pipe

    def suggest(trial):
        return {
            "C":           trial.suggest_float("C",     0.1, 100, log=True),
            "gamma":       trial.suggest_float("gamma", 1e-3, 1e1, log=True),
            "wpow":        trial.suggest_float("wpow", 0.5, 2.5),
            "max_samples": trial.suggest_float("max_samples", 0.6, 1.0),
        }

    def _bag_predict_proba(X_tr_pre, y_tr, Yb_tr, X_te_pre, svm_params, wpow,
                          n_est, max_samples):
        w = np.abs(Yb_tr[:, 1] - Yb_tr[:, 0]) ** wpow + 1e-12
        n = len(X_tr_pre)
        sample_n = int(max_samples * n)
        idx_pos = np.where(y_tr == 1)[0]
        idx_neg = np.where(y_tr == 0)[0]
        n_pos = int(round(sample_n * len(idx_pos) / n))
        n_neg = sample_n - n_pos
        probs = np.zeros((len(X_te_pre), 2))
        for seed in range(n_est):
            rng = np.random.default_rng(seed)
            sel = np.concatenate([rng.choice(idx_pos, size=n_pos, replace=True),
                                  rng.choice(idx_neg, size=n_neg, replace=True)])
            base = SVC(kernel="rbf", probability=True, random_state=seed, **svm_params)
            base.fit(X_tr_pre[sel], y_tr[sel], sample_weight=w[sel])
            probs += base.predict_proba(X_te_pre)
        return probs / n_est

    def fit_predict(pipe, X_tr, y_tr, Yb_tr, X_te):
        pre_pipe = Pipeline(pipe.steps[:-1])
        pre_pipe.fit(X_tr)
        Xs_tr = pre_pipe.transform(X_tr)
        Xs_te = pre_pipe.transform(X_te)

        # Outer inner CV already splits by year; plain KFold here is fine.
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        oof_p = np.zeros(len(X_tr))
        for inner_tr, inner_va in kf.split(Xs_tr):
            p_va = _bag_predict_proba(
                Xs_tr[inner_tr], y_tr[inner_tr], Yb_tr[inner_tr],
                Xs_tr[inner_va],
                pipe._svm_params, pipe._wpow,
                n_est=max(3, pipe._n_estimators // 2),
                max_samples=pipe._max_samples,
            )
            oof_p[inner_va] = p_va[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(oof_p, y_tr.astype(float))

        full_p = _bag_predict_proba(
            Xs_tr, y_tr, Yb_tr, Xs_te,
            pipe._svm_params, pipe._wpow,
            n_est=pipe._n_estimators,
            max_samples=pipe._max_samples,
        )
        p_k1_cal = iso.predict(full_p[:, 1])
        return (p_k1_cal > 0.5).astype(int)

    return Experiment(f"CalibBag-MW/{pre}", build, suggest, n_trials=n_trials,
                      fit_predict=fit_predict)


def decision_theoretic_exp(pre_clf: str = "log_std", pre_reg: str = "log_std",
                           n_trials: int = 60, n_estimators: int = 10) -> Experiment:
    """Combine bagged SVM-MW P(k1) with MO-SVR margin estimate via expected utility.

    Final score per class c = (1-alpha) * P(c) + alpha * normalised(Yb_pred[c]).
    """
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.svm import SVR

    def build(p):
        return {
            "C": p["C"], "gamma": p["gamma"], "wpow": p["wpow"],
            "max_samples": p["max_samples"], "n_estimators": n_estimators,
            "svr_C": p["svr_C"], "svr_gamma": p["svr_gamma"],
            "svr_epsilon": p["svr_epsilon"],
            "alpha": p["alpha"],
        }

    def suggest(trial):
        return {
            "C":           trial.suggest_float("C",     0.5, 50, log=True),
            "gamma":       trial.suggest_float("gamma", 1e-3, 1, log=True),
            "wpow":        trial.suggest_float("wpow", 0.5, 2.5),
            "max_samples": trial.suggest_float("max_samples", 0.6, 1.0),
            "svr_C":       trial.suggest_float("svr_C",     0.5, 50, log=True),
            "svr_gamma":   trial.suggest_float("svr_gamma", 1e-3, 1, log=True),
            "svr_epsilon": trial.suggest_float("svr_epsilon", 1e-3, 0.3, log=True),
            "alpha":       trial.suggest_float("alpha", 0.0, 1.0),
        }

    def fit_predict(params, X_tr, y_tr, Yb_tr, X_te):
        pre_clf_pipe = Pipeline(_pre(pre_clf)); pre_clf_pipe.fit(X_tr)
        Xs_clf_tr = pre_clf_pipe.transform(X_tr)
        Xs_clf_te = pre_clf_pipe.transform(X_te)
        w = np.abs(Yb_tr[:, 1] - Yb_tr[:, 0]) ** params["wpow"] + 1e-12
        n = len(X_tr); sample_n = int(params["max_samples"] * n)
        idx_pos = np.where(y_tr == 1)[0]; idx_neg = np.where(y_tr == 0)[0]
        n_pos = int(round(sample_n * len(idx_pos)/n))
        n_neg = sample_n - n_pos
        clf_probs = np.zeros((len(X_te), 2))
        for seed in range(params["n_estimators"]):
            rng = np.random.default_rng(seed)
            sel = np.concatenate([rng.choice(idx_pos, size=n_pos, replace=True),
                                  rng.choice(idx_neg, size=n_neg, replace=True)])
            base = SVC(kernel="rbf", probability=True, random_state=seed,
                       C=params["C"], gamma=params["gamma"])
            base.fit(Xs_clf_tr[sel], y_tr[sel], sample_weight=w[sel])
            clf_probs += base.predict_proba(Xs_clf_te)
        clf_probs /= params["n_estimators"]

        pre_reg_pipe = Pipeline(_pre(pre_reg)); pre_reg_pipe.fit(X_tr)
        Xs_reg_tr = pre_reg_pipe.transform(X_tr)
        Xs_reg_te = pre_reg_pipe.transform(X_te)
        reg = MultiOutputRegressor(SVR(
            kernel="rbf", C=params["svr_C"], gamma=params["svr_gamma"],
            epsilon=params["svr_epsilon"],
        ), n_jobs=1)
        reg.fit(Xs_reg_tr, Yb_tr)
        Yb_pred = np.clip(reg.predict(Xs_reg_te), 0, None)

        Yb_pred_norm = Yb_pred / (Yb_pred.sum(axis=1, keepdims=True) + 1e-12)
        score = (1 - params["alpha"]) * clf_probs + params["alpha"] * Yb_pred_norm
        return np.argmax(score, axis=1)

    return Experiment(f"DecTheor/{pre_clf}+{pre_reg}", build, suggest,
                      n_trials=n_trials, fit_predict=fit_predict)


def logreg_exp(pre: str = "log_std", n_trials: int = 40) -> Experiment:
    def build(p):
        steps = _pre(pre) + [("model", LogisticRegression(
            C=p["C"], solver="lbfgs", max_iter=3000,
            class_weight=p.get("class_weight"),
        ))]
        return Pipeline(steps)
    def suggest(trial):
        return {
            "C":            trial.suggest_float("C", 1e-3, 1e3, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
    return Experiment(f"LogReg/{pre}", build, suggest, n_trials=n_trials)
