from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    n_samples, n_features = X.shape
    shuffled_indexes = np.arange(n_samples)
    np.random.shuffle(shuffled_indexes)
    folded_array = np.array_split(shuffled_indexes, cv)
    train_err = np.empty(cv)
    valid_err = np.empty(cv)
    for i in range(cv):
        cur_X = np.delete(X, folded_array[i], axis=0)
        cur_y = np.delete(y, folded_array[i])
        estimator.fit(cur_X, cur_y)
        train_pred_y = estimator.predict(cur_X)
        train_err[i] = scoring(cur_y, train_pred_y)
        valid_x = X[folded_array[i]]
        valid_y = y[folded_array[i]]
        y_pred = estimator.predict(valid_x)
        valid_err[i] = scoring(valid_y, y_pred)
    return np.mean(train_err), np.mean(valid_err)





