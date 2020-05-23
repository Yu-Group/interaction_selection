import numpy as np
import matplotlib.pyplot as plt
import irf
from irf.ensemble import wrf as rfc
from irf.utils import (
    get_prevalent_interactions,
    visualize_impurity_decrease,
    visualize_prevalent_interactions,
    get_filtered_feature_paths
)
import rpy2.robjects as robjects
from irf.irf_jupyter_utils import draw_tree


def load_data(
    i=0,
    name="Sim",
    rule="and",
):
    if name == "Sim":
        if rule == "and":
            robjects.r['load']("../../signediRF/data/gaussSim_and.Rdata")
            interact_true = [
                [(0, 'R'), (1, 'R'), (2, 'R'), (3, 'R')],
            ]
        elif rule == "or":
            robjects.r['load']("../../signediRF/data/gaussSim_or.Rdata")
            interact_true = [
                [(0, 'R'), (1, 'R'), (2, 'L'), (3, 'L')],
                [(0, 'L'), (1, 'L'), (2, 'R'), (3, 'R')],
            ]
        elif rule == "add":
            robjects.r['load']("../../signediRF/data/gaussSim_add.Rdata")
            interact_true = [
                [(0, 'R'), (1, 'R'), (2, 'R')],
                [(3, 'R'), (4, 'R'), (5, 'R')],
            ]
        else:
            raise ValueError(
                "rule (%s) is not allowed. only take and, or, add"
            )
        X = np.array(robjects.r['data'][i])[:-5000].reshape((50, 5000)).T
        y = np.array(robjects.r['data'][i])[-5000:]
    elif name == "Enhancer":
        if rule == "and":
            robjects.r['load']("../../signediRF/data/enhancer_sim_and.Rdata")
            interact_true = [
                [(45, "R"), (49, "R"), (63, "R"), (74, "R")]
            ]
        elif rule == "or":
            robjects.r['load']("../../signediRF/data/enhancer_sim_or.Rdata")
            interact_true = [
                [(45, "R"), (74, "R"), (49, "L"), (63, "L")],
                [(45, "L"), (74, "L"), (49, "R"), (63, "R")],
            ]
        elif rule == "add":
            robjects.r['load']("../../signediRF/data/enhancer_sim_add.Rdata")
            interact_true = [
                [(49, "R"), (53, "R"), (63, "R")],
                [(45, "R"), (74, "R"), (77, "R")],
            ]
        else:
            raise ValueError(
                "rule (%s) is not allowed. only take and, or, add"
            )
        raw = np.array(robjects.r['data'][i])
        len_train_id = int(raw[-1])
        train_id = raw[-(len_train_id + 1):-1].astype(int) - 1
        X = np.array(robjects.r['data'][i])[:7809 * 80].reshape((80, 7809)).T
        y = np.array(robjects.r['data'][i])[7809*80:7809*81]
        X = X[train_id]
        y = y[train_id]
    else:
        raise ValueError("name is not allowed. only Enhancer or Sim")
    return X, y, interact_true


def train_model(X, y, name="Sim", rule="and", weight_scheme="depth"):
    if name == "Sim":
        rf = rfc(
            n_jobs=4,
            n_estimators=100,
            bootstrap=False,
        )
    else:
        raise ValueError("name not allowed")
    rf.fit(
        X,
        y,
        keep_record=False,
        K=10,
    )
    if name == "Sim":
        min_support = 500 #if rule == "or" else 500
    else:
        min_support = 500
    prevalence = get_prevalent_interactions(
        rf,
        impurity_decrease_threshold=1e-3,
        min_support=min_support,
        signed=True,
        weight_scheme=weight_scheme,
    )
    return list(prevalence.keys())

def fit_score(interact_pred, interact_true):
    """
    Compute the score of two sets
    We ignore the direction in this problem.
    """
    if len(interact_pred) == 0:
        return 0
    if isinstance(interact_pred[0], tuple):
        interact_pred = set([x[0] for x in interact_pred])
    if isinstance(interact_true[0], tuple):
        interact_true = set([x[0] for x in interact_true])
    if interact_pred.issubset(interact_true):
        return len(interact_pred) / len(interact_true)
    else:
        return 0

def evaluate_model(y_pred, y_true):
    """
    Evaluate the predicted interactions.
    Each prediction gets a score for the best match for each elem in y_true
    For example, y_pred = [(1+, 2+),(1+,4+)], y_true = [(1-, 2+, 3+)],
    then using the score is 2/3.

    Parameters
    ----------
    y_pred : list, the predicted interactions

    y_true : list, the real interactions


    Returns
    -------
    avg_scores : list that has the same length as y_pred
        avg_scores[i] is the highest score of first i+1 elems in matching
        y_true.
    """
    avg_scores = np.zeros((len(y_pred),))
    for interact_true in y_true:
        scores = [
            fit_score(interact_pred, interact_true) for interact_pred in y_pred
        ]
        for i in range(1, len(scores)):
            scores[i] = max(scores[i], scores[i-1])
        avg_scores += np.array(scores)
    avg_scores /= len(y_true)
    return avg_scores


if __name__ == "__main__":
    for name in ["Sim", "Enhancer"]:
        for rule in ["and", "or", "add"]:
            for i in range(50):
                X, y = load_data(i=i, name=name, rule=rule)
    print("load_data sucess.")
