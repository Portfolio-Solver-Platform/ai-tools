"""Model specs for the portfolio-selection comparison."""
from __future__ import annotations

import numpy as np
import optuna
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb

from .harness import ModelSpec


def _scaled(estimator):
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def dummy_spec() -> ModelSpec:
    def build(p):
        return DummyClassifier(strategy="most_frequent")
    def suggest(trial):
        return {}
    return ModelSpec("Dummy", build, suggest, n_trials=1)


def logreg_spec() -> ModelSpec:
    def build(p):
        return _scaled(LogisticRegression(
            C=p["C"], solver="lbfgs", max_iter=2000,
        ))
    def suggest(trial):
        return {"C": trial.suggest_float("C", 1e-3, 1e3, log=True)}
    return ModelSpec("LogReg", build, suggest, n_trials=30)


def knn_spec() -> ModelSpec:
    def build(p):
        return _scaled(KNeighborsClassifier(
            n_neighbors=p["k"], weights=p["weights"], n_jobs=1,
        ))
    def suggest(trial):
        return {
            "k":       trial.suggest_int("k", 1, 50),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        }
    return ModelSpec("kNN", build, suggest, n_trials=30)


def svm_spec() -> ModelSpec:
    def build(p):
        return _scaled(SVC(kernel="rbf", C=p["C"], gamma=p["gamma"]))
    def suggest(trial):
        return {
            "C":     trial.suggest_float("C",     0.1, 100, log=True),
            "gamma": trial.suggest_float("gamma", 1e-3, 1e1, log=True),
        }
    return ModelSpec("SVM-RBF", build, suggest, n_trials=50, expensive=True)


def rf_spec() -> ModelSpec:
    def build(p):
        return RandomForestClassifier(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            min_samples_leaf=p["min_samples_leaf"],
            max_features=p["max_features"],
            n_jobs=1, random_state=42,
        )
    def suggest(trial):
        return {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
            "max_depth":        trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 1.0]),
        }
    return ModelSpec("RandomForest", build, suggest, n_trials=40)


def extratrees_spec() -> ModelSpec:
    def build(p):
        return ExtraTreesClassifier(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            min_samples_leaf=p["min_samples_leaf"],
            max_features=p["max_features"],
            n_jobs=1, random_state=42,
        )
    def suggest(trial):
        return {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
            "max_depth":        trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 1.0]),
        }
    return ModelSpec("ExtraTrees", build, suggest, n_trials=40)


def xgb_spec() -> ModelSpec:
    def build(p):
        return xgb.XGBClassifier(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"],
            min_child_weight=p["min_child_weight"],
            tree_method="hist", n_jobs=1, random_state=42, verbosity=0,
        )
    def suggest(trial):
        return {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        }
    return ModelSpec("XGBoost", build, suggest, n_trials=50)


def lgbm_spec() -> ModelSpec:
    def build(p):
        return lgb.LGBMClassifier(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            num_leaves=p["num_leaves"],
            learning_rate=p["learning_rate"],
            subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"],
            min_child_samples=p["min_child_samples"],
            n_jobs=1, random_state=42, verbose=-1,
        )
    def suggest(trial):
        return {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        }
    return ModelSpec("LightGBM", build, suggest, n_trials=50)


ALL_SPECS = [
    dummy_spec(),
    logreg_spec(),
    knn_spec(),
    svm_spec(),
    rf_spec(),
    extratrees_spec(),
    xgb_spec(),
    lgbm_spec(),
]
