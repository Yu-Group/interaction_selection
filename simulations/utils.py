import numpy as np
import matplotlib.pyplot as plt
import irf
import pandas as pd
from irf.ensemble import wrf as rfc
from irf.ensemble import wrf_reg as rfr
from irf.utils import (
    get_prevalent_interactions,
    visualize_impurity_decrease,
    visualize_prevalent_interactions,
    get_filtered_feature_paths
)
from irf.irf_jupyter_utils import draw_tree
from os.path import join as oj
import os
from sklearn.mixture import GaussianMixture

def load_siRF_result(
    i=0,
    name="Sim",
    rule="and",
):
    if name == "Sim":
        data = pd.read_csv("~/sim_{}.csv".format(rule))
        if rule == "and":
            interact_true = [
                [(0, 'R'), (1, 'R'), (2, 'R'), (3, 'R')],
            ]
        elif rule == "or":
            interact_true = [
                [(0, 'R'), (1, 'R'), (2, 'L'), (3, 'L')],
                [(0, 'L'), (1, 'L'), (2, 'R'), (3, 'R')],
            ]
        elif rule == "add":
            interact_true = [
                [(0, 'R'), (1, 'R'), (2, 'R')],
                [(3, 'R'), (4, 'R'), (5, 'R')],
            ]
        else:
            raise ValueError(
                "rule (%s) is not allowed. only take and, or, add"
            )
        raw_list = data.loc[data['rep'] == i, 'int']
        y_pred = [[(int(x[1:-1])-1, 'R') if x[-1] == '+' else (int(x[1:-1])-1, 'L') for x in elem.split("_")] for elem in raw_list]
    elif name == "Enhancer_new":
        data = pd.read_csv("~/enhancer_{}.csv".format(rule))
        if rule == "and":
            interact_true = [
                [(0, "R"), (4, "R"), (18, "R"), (29, "R")]
            ]
        elif rule == "or":
            interact_true = [
                [(4, "R"), (18, "R"), (0, "L"), (29, "L")],
                [(4, "L"), (18, "L"), (0, "R"), (29, "R")],
            ]
        elif rule == "add":
            interact_true = [
                [(4, "R"), (8, "R"), (18, "R")],
                [(0, "R"), (29, "R"), (32, "R")],
            ]
        else:
            raise ValueError(
                "rule (%s) is not allowed. only take and, or, add"
            )
        raw_list = data.loc[data['rep'] == i, 'int']
        y_pred = [[(x[:-1].lower() if len(x) > 2 else x[:-1], 'R') if x[-1] == '+' else (x[:-1].lower() if len(x) > 2 else x[:-1], 'L') for x in elem.split("_")] for elem in raw_list]
    else:
        raise ValueError("name is not allowed. only Enhancer or Sim")
    return y_pred, interact_true

def load_simulated_regression_data(
    i=0,
    num_interact=2,
    SNR=100,
    order=2,
    region=0.5,
    p=50,
    n=5000,
):
    '''
    load simulated regression data with given parameteres
    
    Parameters
    ----------
    i : the random seed
    num_iteract : number of true interactions
    SNR :  signal to noise ratio
        defined as var(signal) / var(noise), note that the outcome = signal + noise
    order : the order of interactions
    region : the size of the region covered by interactions
        this measures the number of samples that fall into at least one of the rules
    p : the number of dimensions
    n : the number of samples
    
    Returns
    -------
    X : data matrix
    y : response
    interact_true : the true interactions
    '''
    np.random.seed(i+100)
    interact_true = [
        [(ind, 'L') for ind in range(start * order, (start+1) * order)] for start in range(num_interact)
    ]
    X = np.random.uniform(size=(n, p))
    # formula: region = 1 - (1 - threshold ** order) ** num_interact
    threshold = (1 - (1 - region) ** (1/num_interact)) ** (1/order)
    def f(X):
        out = np.zeros_like(X[:,0]) * 1
        for start in range(num_interact):
            out += X[:,start*order:((start+1)*order)].prod(axis=1) * 1
        return out
    signal = f(X[:,:order * num_interact] < threshold)
    noise = (np.var(signal) / SNR) ** .5 * np.random.normal(size=signal.shape)
    y = signal + noise
    return X, y, interact_true

def load_data(
    i=0,
    name="Sim",
    rule="and",
):
    import rpy2.robjects as robjects
    if name == "Sim":
        if rule == "and":
            robjects.r['load']("../data/gaussSim_and.Rdata")
            interact_true = [
                [(0, 'R'), (1, 'R'), (2, 'R'), (3, 'R')],
            ]
        elif rule == "or":
            robjects.r['load']("../data/gaussSim_or.Rdata")
            interact_true = [
                [(0, 'R'), (1, 'R'), (2, 'L'), (3, 'L')],
                [(0, 'L'), (1, 'L'), (2, 'R'), (3, 'R')],
            ]
        elif rule == "add":
            robjects.r['load']("../data/gaussSim_add.Rdata")
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
            robjects.r['load']("../data/enhancer_sim_and.Rdata")
            interact_true = [
                [(45, "R"), (49, "R"), (63, "R"), (74, "R")]
            ]
        elif rule == "or":
            robjects.r['load']("../data/enhancer_sim_or.Rdata")
            interact_true = [
                [(45, "R"), (74, "R"), (49, "L"), (63, "L")],
                [(45, "L"), (74, "L"), (49, "R"), (63, "R")],
            ]
        elif rule == "add":
            robjects.r['load']("../data/enhancer_sim_add.Rdata")
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
    elif name == "Enhancer_new":
        if rule == "and":
            robjects.r['load']("../data/enhancer_new_sim_and.Rdata")
            interact_true = [
                [(0, "R"), (4, "R"), (18, "R"), (29, "R")]
            ]
        elif rule == "or":
            robjects.r['load']("../data/enhancer_new_sim_or.Rdata")
            interact_true = [
                [(4, "R"), (18, "R"), (0, "L"), (29, "L")],
                [(4, "L"), (18, "L"), (0, "R"), (29, "R")],
            ]
        elif rule == "add":
            robjects.r['load']("../data/enhancer_new_sim_add.Rdata")
            interact_true = [
                [(4, "R"), (8, "R"), (18, "R")],
                [(0, "R"), (29, "R"), (32, "R")],
            ]
        else:
            raise ValueError(
                "rule (%s) is not allowed. only take and, or, add"
            )
        raw = np.array(robjects.r['data'][i])
        len_train_id = int(raw[-1])
        train_id = raw[-(len_train_id + 1):-1].astype(int) - 1
        X = np.array(robjects.r['data'][i])[:7809 * 35].reshape((35, 7809)).T
        y = np.array(robjects.r['data'][i])[7809*35:7809*36]
        X = X[train_id]
        y = y[train_id]
    elif name == 'Sim_new':
        np.random.seed(i * 100)
        n, p = 5000, 50
        X = np.random.normal(0, 1, size=(n, p))
        if rule == 'single':
            n_features = (i//20 + 4)
            y = 0.8 * (X[:,:n_features] < 0).all(axis=1) + 0.05
            for j in range(n):
                y[j] = np.random.choice([0,1], p=(1 - y[j], y[j]))
            interact_true = [
                [(j, "L") for j in range(n_features)],
            ]
        elif rule == 'multiple':
            n_features = (i//20) * 2 + 2
            rules = 1 * (X[:,:n_features] < 0)
            rules = rules[:,:n_features//2] * rules[:,n_features//2:]
            y = (rules).sum(axis=1) * 0.1 + 0.05
            for j in range(n):
                y[j] = np.random.choice([0,1], p=(1 - y[j], y[j]))
            interact_true = [
                [(j, "L"), (j+n_features//2, "L")] for j in range(n_features//2)
            ]
        elif rule == 'overlap':
            rule1 = 1 * (X[:,:6] < 0)
            rule2 = 1 * (X[:,(6-(i//20)):(12-(i//20))] > 0)
            y = (rule1).all(axis=1) * 0.4 + (rule2).all(axis=1) * 0.4 + 0.05
            for j in range(n):
                y[j] = np.random.choice([0,1], p=(1 - y[j], y[j]))
            interact_true = [
                [(j, "L") for j in range(6)],
                [(j + 10 - i//5, "R") for j in range(6)],
            ]
    else:
        raise ValueError("name is not allowed. only Enhancer or Sim")
    return X, y, interact_true

def cache_result(filename, attribute_dict, results, dir='.'):
    '''
    This function is used to cache the results of the models
    '''
    filename = oj(dir, filename)
    if os.path.exists(filename):
        prev = pd.read_csv(filename, index_col=None)
    else:
        prev = pd.DataFrame({'attributes':[], 'results':[]})
    out = prev.append({'attributes':attribute_dict, 'results':results}, ignore_index=True)
    out.to_csv(filename, index=False)
    return
    
def find_results_from_cache(filename, attribute_dict, dir='.'):
    '''
    This function tries to find the result given attribute_dict
    
    '''
    filename = oj(dir, filename)
    try:
        cache = pd.read_csv(filename, index_col=None)
    except:
        print("cannot load file {}, quit.".format(filename))
        return
    result = None
    for i in reversed(range(cache.shape[0])):
        if cache.iloc[i]['attributes'] == str(attribute_dict):
            result = cache.iloc[i]['results']
            break
    try:
        result = eval(result)
    except:
        pass
    return result
def train_regression_model(
    X,
    y,
    weight_scheme="depth",
    intended_order=4,
    bootstrap=True,
    feature_selection='hard',
    n_jobs=16,
    tag="regression_model_results"
):
    if feature_selection == 'hard':
        rf = rfr(
            n_jobs=n_jobs,
            n_estimators=100,
            max_depth=None,
            bootstrap=bootstrap,
        )
        rf.fit(
            X,
            y,
            keep_record=False,
            K=9,
        )
        gm = GaussianMixture(
            n_components=2,
            means_init=[[np.min(rf.feature_importances_)], [np.max(rf.feature_importances_)]],
        )
        selected_features = gm.fit_predict(rf.feature_importances_.reshape((-1,1)))
        new_importance = gm.predict_proba(rf.feature_importances_.reshape((-1,1)))[:,1].flatten() #* rf.feature_importances_
        rf = rfr(
            n_jobs=n_jobs,
            n_estimators=300,
            max_features=int(np.sum(selected_features) ** .5),
            max_depth=None,
            bootstrap=bootstrap,
        )
        rf.fit(
            X,
            y,
            keep_record=False,
            feature_weight=new_importance,
            K=1,
        )
    elif feature_selection == 'soft':
        rf = rfr(
            n_jobs=n_jobs,
            n_estimators=300,
            max_depth=None,
            bootstrap=bootstrap,
        )
        rf.fit(
            X,
            y,
            keep_record=False,
            K=10,
        )
    min_support = rf.n_paths // 2 ** (intended_order + 1)
    prevalence = get_prevalent_interactions(
        rf,
        impurity_decrease_threshold=1e-3,
        min_support=min_support,
        signed=True,
        weight_scheme=weight_scheme,
        adjust_for_weights=True,
    )
    return list(prevalence.keys())

def train_model(
    X,
    y,
    iter,
    name="Sim",
    rule="and",
    weight_scheme="depth",
    bootstrap=True,
    tag='DefaultTag',
    use_cache=True,
    cache_after_run=True,
):
    params =  {
        'name':name,
        'rule':rule,
        'weight_scheme':weight_scheme,
        'bootstrap':bootstrap,
        'iter':iter,
    }
    if weight_scheme == 'siRF':
        # load siRF result
        y_pred, y_true = load_siRF_result(i=iter+1, name=name, rule=rule)
        return y_pred
    # load cache first
    if use_cache:
        cache = find_results_from_cache(tag + '.csv', params)
        if cache is not None:
            return [x[0] for x in cache]
    rf = rfc(
        n_jobs=4,
        n_estimators=100,
        max_depth=None,
        bootstrap=bootstrap,
    )
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
    if name == 'Enhancer_new':
        mask = ["zld", "bcd", "bcd", "cad", "D", "da", "dl", "ftz", "gt", "h",
         "h", "hb", "hb", "hkb", "hkb", "hkb", "kni", "kni", "kr", "kr", 
         "mad", "med", "prd", "prd", "run", "run", "shn", "shn", "slp1", "sna", 
         "sna", "tll", "twi", "twi", "z"]
        mask = {x:y for x, y in enumerate(mask)}
    else:
        mask = None
    prevalence = get_prevalent_interactions(
        rf,
        impurity_decrease_threshold=1e-3,
        min_support=min_support,
        signed=True,
        weight_scheme=weight_scheme,
        mask=mask,
    )
    if cache_after_run:
        cache_result(tag+'.csv', params, list(prevalence.items()))
    return list(prevalence.keys())

def fit_score(interact_pred, interact_true, type='strict', with_sign=False):
    """
    Compute the score of two sets
    We ignore the direction in this problem.
    
    type: ['strict', 'medium', 'mild']
        if type is 'strict', 
            score = 0 if S_pred is not subset of S_true and |S_pred| / |S_true| otherwise
        elif type is 'medium':
            score = 0 if S_pred is not subset of S_true and 1 otherwise
        elif type is 'mild':
            score = Jaccard distance(S_pred, S_true)
    """
    if len(interact_pred) == 0:
        return 0
    if not with_sign and isinstance(interact_pred[0], tuple):
        interact_pred = set([x[0] for x in interact_pred])
    else:
        interact_pred = set(interact_pred)
    if not with_sign and isinstance(interact_true[0], tuple):
        interact_true = set([x[0] for x in interact_true])
    else:
        interact_true = set(interact_true)
    if type == 'strict':
        if interact_pred.issubset(interact_true):
            return len(interact_pred) / len(interact_true)
        else:
            return 0
    elif type == 'medium':
        if interact_pred.issubset(interact_true):
            return 1
        else:
            return 0
    elif type == 'mild':
        return len(interact_pred.intersection(interact_true)) / len(interact_pred.union(interact_true))
    else:
        raise ValueError('type({}) is not acceptable.'.format)

def evaluate_model(y_pred, y_true, name, metric, with_sign=False):
    """
    Evaluate the predicted interactions.
    Each prediction gets a score for the best match for each elem in y_true
    For example, y_pred = [(1+, 2+),(1+,4+)], y_true = [(1-, 2+, 3+)],
    then using the score is 2/3.

    Parameters
    ----------
    y_pred : list, the predicted interactions

    y_true : list, the real interactions
    
    name : str, the name of the simulation ["Enhancer_new", "Enhancer", "Sim"]
        When name is enhancer_new, features are converted to genes

    metric : str, ['strict', 'medium', 'mild']
    
    with_sign:

    Returns
    -------
    avg_scores : list that has the same length as y_pred
        avg_scores[i] is the highest score of first i+1 elems in matching
        y_true.
    """
    if name == 'Enhancer_new':
        mask = ["zld", "bcd", "bcd", "cad", "D", "da", "dl", "ftz", "gt", "h",
         "h", "hb", "hb", "hkb", "hkb", "hkb", "kni", "kni", "kr", "kr", 
         "mad", "med", "prd", "prd", "run", "run", "shn", "shn", "slp1", "sna", 
         "sna", "tll", "twi", "twi", "z"]
        mask = {x:y for x, y in enumerate(mask)}
        y_true = [[(mask[x[0]], x[1]) for x in interact] for interact in y_true]
        #y_pred = [[(mask[x[0]], x[1]) for x in interact] for interact in y_pred]
    avg_scores = np.zeros((len(y_pred),))
    for interact_true in y_true:
        scores = [
            fit_score(interact_pred, interact_true, type=metric, with_sign=with_sign) for interact_pred in y_pred
        ]
        for i in range(1, len(scores)):
            scores[i] = max(scores[i], scores[i-1])
        avg_scores += np.array(scores)
    avg_scores /= len(y_true)
    return avg_scores


if __name__ == "__main__":
    for name in ["Sim", "Enhancer", "Enhancer_new"]:
        for rule in ["and", "or", "add"]:
            for i in range(50):
                X, y = load_data(i=i, name=name, rule=rule)
    print("load_data sucess.")
