import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# Global solver order: (cpsat8, k1, ek1).
# LABEL_MAPS[name][local_class] = global solver index
LABEL_MAPS = {
    'cpsat8_k1_ek1': [0, 1, 2],
    'cpsat8_k1':     [0, 1],
    'cpsat8_ek1':    [0, 2],
}


def make_train_val_test_indices(groups, test_size=0.2, val_size=0.25, random_state=42):
    """Group-by-problem train/val/test split.

    First splits off a test set (test_size of all data), then splits the remaining
    train+val portion into train and val (val_size is the fraction of train+val).
    With defaults, the result is 60/20/20 (train/val/test) of the original data.
    """
    n = len(groups)
    all_idx = np.arange(n)
    s1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trainval_idx, test_idx = next(s1.split(all_idx, groups=groups))
    s2 = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    rel_train, rel_val = next(s2.split(trainval_idx, groups=groups[trainval_idx]))
    return trainval_idx[rel_train], trainval_idx[rel_val], test_idx


def leave_one_year_out_folds(years):
    """Yield (year, train_idx, test_idx) for each unique year, chronologically.

    Used as the outer loop of nested CV: each year takes a turn as the held-out
    test set, with the other 14 years available for HPO + refit.
    """
    folds = []
    for y in sorted(np.unique(years).tolist()):
        test_idx = np.where(years == y)[0]
        train_idx = np.where(years != y)[0]
        folds.append((int(y), train_idx, test_idx))
    return folds


def year_kfold_folds(years, n_splits=5):
    """K-fold over years: yields (label, train_idx, test_idx) for n_splits folds.

    Uses sklearn GroupKFold so years are balanced across folds (with 15 years
    and n_splits=5, each fold's test set contains exactly 3 years). Label is
    a short description of the years held out, useful for logging.

    Cheaper than leave_one_year_out_folds when an approximation is enough.
    """
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=n_splits)
    idx = np.arange(len(years))
    folds = []
    for tr, te in gkf.split(idx, groups=years):
        held = sorted(set(int(y) for y in years[te]))
        label = "-".join(str(y) for y in held)
        folds.append((label, tr, te))
    folds.sort(key=lambda f: f[0])
    return folds


def map_to_global(predictions_local, dataset_name):
    return np.array(LABEL_MAPS[dataset_name])[predictions_local]


def shared_test_borda(predictions_local, dataset_name, Y_test_borda_3solver):
    pred_global = map_to_global(predictions_local, dataset_name)
    return np.sum(Y_test_borda_3solver[np.arange(len(Y_test_borda_3solver)), pred_global])
