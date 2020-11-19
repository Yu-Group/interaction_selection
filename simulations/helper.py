from utils import (
    train_regression_model,
    train_regression_model_iRF,
    evaluate_model,
    load_simulated_regression_data,
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from irf.ensemble import wrf_reg as rfr
from irf.tree import WeightedDecisionTreeRegressor as wtr
from irf.ensemble import RandomForestRegressorWithWeights as wrfr
import pickle as pk

def ff(name, num_interact, order, SNR, ind):
    final1 = {}
    final2 = {}
    X, y, y_true = load_simulated_regression_data(i=ind,SNR=SNR, num_interact=num_interact, order=order, n=1000, p = 50)
    for feature_selection in ['soft', 'hard']:
        y_pred = train_regression_model(X, y, feature_selection=feature_selection, intended_order=order, threshold=0, n_jobs=1)
        for metric in ['strict', 'mild']:
            result = evaluate_model(y_pred, y_true, name, metric=metric, with_sign=True)
            final1[(name, num_interact, order, SNR, ind, feature_selection, metric)] = result
            final2[(name, num_interact, order, SNR, ind, feature_selection, metric)] = [X, y, y_true, y_pred]
    return final1, final2

def ff_2(name, num_interact, order, SNR, feature_correlation, overlap, noise_type, ind):
    final1 = {}
    final2 = {}
    X, y, y_true = load_simulated_regression_data(
        i=ind,
        SNR=SNR,
        num_interact=num_interact,
        order=order,
        n=1000,
        p = 50,
        feature_correlation=feature_correlation,
        overlap=overlap,
        noise_type=noise_type,
    )
    for feature_selection in ['soft', 'hard']:
        y_pred = train_regression_model(
            X,
            y,
            feature_selection=feature_selection,
            intended_order=order,
            threshold=0,
            n_jobs=1)
        for metric in ['strict', 'mild']:
            result = evaluate_model(y_pred, y_true, name, metric=metric, with_sign=True)
            final1[(feature_correlation, overlap, noise_type, ind, feature_selection, metric)] = result
            final2[(feature_correlation, overlap, noise_type, ind, feature_selection, metric)] = [X, y, y_true, y_pred]
    return final1, final2

def ff_3(name, num_interact, order, SNR, feature_correlation, overlap, noise_type, ind):
    final1 = {}
    final2 = {}
    X, y, y_true = load_simulated_regression_data(
        i=ind,
        SNR=SNR,
        num_interact=num_interact,
        order=order,
        n = 1000,
        p = 50,
        feature_correlation=feature_correlation,
        overlap=overlap,
        noise_type=noise_type,
    )
    y_pred = train_regression_model_iRF(
        X,
        y,
    )
    for metric in ['strict', 'mild']:
        result = evaluate_model(y_pred, y_true, name, metric=metric, with_sign=False)
        final1[(feature_correlation, overlap, noise_type, ind, 'soft', metric)] = result
        final2[(feature_correlation, overlap, noise_type, ind, 'soft', metric)] = [X, y, y_true, y_pred]
    return final1, final2
