from __future__ import annotations
from typing import NoReturn, Tuple
from ...base import BaseEstimator
import numpy as np
from ...metrics import misclassification_error
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        n_samples, n_features = X.shape
        feature_thresh = X[0]
        label_sign = np.sign(feature_thresh)
        loss = np.full(n_features, n_samples, dtype=float)
        for f in range(n_features):
            values = X[:,f]
            f_threshs_one = self._find_threshold(values, y, 1)
            f_threshs_minus = self._find_threshold(values, y, -1)
            feature_thresh[f] = f_threshs_minus[0]
            loss[f] = f_threshs_minus[1]
            label_sign[f] = -1
            if f_threshs_one[1] < f_threshs_minus[1]:
                feature_thresh[f] = f_threshs_one[0]
                loss[f] = f_threshs_one[1]
                label_sign[f] = 1
        self.j_ = np.argmin(loss)
        self.sign_ = label_sign[self.j_]
        self.threshold_ = feature_thresh[self.j_]
        self.fitted_ = True

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        feature_vals = X[:,self.j_]
        y_pred = np.full(len(feature_vals), self.sign_)
        neg_indexes = np.argwhere(feature_vals < self.threshold_)
        y_pred[neg_indexes] = (-1) * self.sign_
        return y_pred



    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        n_samples = len(values)
        sorted_args = np.argsort(values)
        sorted_values = values[sorted_args]
        sorted_labels = np.sign(labels[sorted_args])
        sorted_D = labels[sorted_args] / np.sum(np.abs(labels))
        F = np.sum(np.abs(sorted_D[np.argwhere(sorted_labels != sign)]))
        loss = F
        thresh = sorted_values[0]
        for thresh_opt in range(1,len(sorted_values)):
            F = F + (sign)*sorted_D[thresh_opt-1]
            if (F < loss):
                loss = F
                thresh = sorted_values[thresh_opt]
        F = F + (sign)*sorted_D[-1]
        if(loss > F):
            loss = F
            thresh = sorted_values[-1]+ 0.000001
        # F = F +(sign)*sorted_labels[-1]
        # if (F < loss):
        #     loss = F
        #     thresh = (sorted_values[-1] + 0.0001)
        return thresh, loss



    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self._predict(X)
        pred_err = np.argwhere(np.sign(y) != y_pred)
        loss = np.sum(np.abs(y[pred_err]))
        return loss
