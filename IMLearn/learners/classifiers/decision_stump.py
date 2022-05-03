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
        # classes, count = np.unique(y, return_counts=True)
        # n_samples, n_features = X.shape

        #https://sefiks.com/2018/08/27/a-step-by-step-cart-decision-tree-example/



    #https://programmer.group/python-implementation-of-cart-decision-tree-algorithm-detailed-comments.html
        #getting first class indexes
        # y_one = np.argwhere(y == classes[0])
        # y_two = np.argwhere(y == classes[1])
        # best_gini = 1
        # best_split_point = 0
        # j_of_best_feature = -1
        # X_trans = X.T
        # for i in range(n_features):
        #     feature_vals = np.unique(X_trans[i])
        #     Gini = {}
        #     for val in feature_vals:
        #         left_of_val = np.argwhere(X_trans[i] >= val)
        #         right_of_val = np.argwhere(X_trans[i] < val)
        #         prob1 = len(left_of_val) / float(n_samples)
        #         prob2 = len(right_of_val) / float(n_samples)
        #         classes1, count1 = np.unique(y[left_of_val], return_counts=True)
        #         classes2, count2 = np.unique(y[right_of_val], return_counts=True)
        #         gini_left = 1 - np.sum(np.power(count1, 2))
        #         gini_right = 1 - np.sum(np.power(count2, 2))
        #         Gini[val] = (prob1 * gini_left) + (prob2 * gini_right)
        #         if Gini[val] < best_gini:
        #             best_gini = Gini[val]
        #             self.j_ = i
        #             self.threshold_ = val
        #             self.sign_ = np.sum(left_of_val) #get positive or negative
        # self.fitted_ = True

        # classes, count = np.unique(y, return_counts=True)
        n_samples, n_features = X.shape
        y_sign = np.sign(y)
        y_one_index = np.argwhere(y_sign == 1)
        y_minus_index = np.argwhere(y_sign == -1)
        one_amount = len(y_one_index)
        minus_amount = len(y_minus_index)
        feature_thresh = X[0]
        label_sign = np.sign(feature_thresh)
        loss = np.full(n_features, n_samples, dtype=float) #loss is number of samples at start
        for f in range(n_features):
            column = X[:,f] #feature column
            f_threshs = self.__fitting_thres_search(column, y)
            feature_thresh[f] = f_threshs[0]
            loss[f] = f_threshs[1]
            label_sign[f] = f_threshs[2]
            # if (loss[f] > 0.5): #of the sign should be the opossite
            #     label_sign[f] = -1
        self.j_ = np.argmin(loss)
        self.sign_ = label_sign[self.j_]
        self.threshold_ = feature_thresh[self.j_]
        self.fitted_ = True


#helper function i added
    def __fitting_thres_search(self, values: np.ndarray, labels: np.ndarray) -> np.ndarray:
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

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        sign: int -1 or 1

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        n_samples = len(values)
        loss = n_samples
        thresh = values[0]
        sign = np.sign(thresh)
        feature_thresh_options = set(values)
        for thresh_opt in feature_thresh_options:
            left = np.argwhere(values >= thresh_opt)
            right = np.argwhere(values < thresh_opt)
            l = labels[left]
            left_sign = np.sign(labels[left]).flatten()
            summm = np.sign(np.sum(left_sign))
            left_sign = summm
            if (left_sign == 0):  # if same amount of both labels
                left_sign = 1
            y_pred = np.full(len(values), (-1)*left_sign)
            y_pred[left] = left_sign
            # curr_loss = np.sum(np.sign(labels) != y_pred) / n_samples
            pred_err = np.argwhere(np.sign(labels) != y_pred).flatten()
            # TODO: is normalised, check
            curr_loss = np.sum(np.abs(labels[pred_err]))
            # curr_loss = np.sum(np.sign(labels[pred_err])) / n_samples
            # pred_err = np.argwhere(np.sign(labels[right]) == y_pred)
            # curr_loss += np.sum(labels[pred_err])
            # curr_loss = ((y[left] != left_sign).sum() + (
            #             y[right] == left_sign).sum()) /len(values) #miss_class_error
            if (curr_loss < loss):
                loss = curr_loss
                thresh = thresh_opt
                sign = left_sign
        return np.array([thresh, loss, sign])


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
        # X_trans = X.T
        # n_samples, n_features = X.shape
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
        loss = n_samples
        thresh = values[0]
        feature_thresh_options = set(values)
        for thresh_opt in feature_thresh_options:
            left = np.argwhere(values >= thresh_opt)
            right = np.argwhere(values < thresh_opt)
            # left_sign = np.sign(sum(y[left]))
            # if (left_sign == 0):  # if same amount of both labels
            #     left_sign = 1
            y_pred = np.full(len(values), (-1)*sign)
            y_pred[left] = sign
            pred_err = np.argwhere(np.sign(labels) != y_pred).flatten()
            # TODO: is normalised, check
            curr_loss = np.sum(np.abs(labels[pred_err]))
            # curr_loss = misclassification_error(labels, y_pred)
            # curr_loss = ((y[left] != left_sign).sum() + (
            #             y[right] == left_sign).sum()) /len(values) #miss_class_error
            if (curr_loss < loss):
                loss = curr_loss
                thresh = thresh_opt
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
        n_samples = len(X)
        y_pred = self._predict(X)
        pred_err = np.argwhere(np.sign(y) != y_pred).flatten()
        #TODO: is normalised, check
        loss = np.sum(np.abs(y[pred_err])) / n_samples
        # loss = misclassification_error(y, y_pred)
        return loss
