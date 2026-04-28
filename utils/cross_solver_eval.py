import numpy as np

# Global solver order: (cpsat8, k1, ek1).
# LABEL_MAPS[name][local_class] = global solver index
LABEL_MAPS = {
    'cpsat8_k1_ek1': [0, 1, 2],
    'cpsat8_k1':     [0, 1],
    'cpsat8_ek1':    [0, 2],
}


def map_to_global(predictions_local, dataset_name):
    return np.array(LABEL_MAPS[dataset_name])[predictions_local]


def shared_test_borda(predictions_local, dataset_name, Y_test_borda_3solver):
    pred_global = map_to_global(predictions_local, dataset_name)
    return np.sum(Y_test_borda_3solver[np.arange(len(Y_test_borda_3solver)), pred_global])
