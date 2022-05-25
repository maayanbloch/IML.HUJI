from __future__ import annotations
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from ...metrics import loss_functions


#TODO: check if allowed to import
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression


class RidgeRegression(BaseEstimator):
    """
    Ridge Regression Estimator

    Solving Ridge Regression optimization problem
    """

    def __init__(self, lam: float, include_intercept: bool = True) -> RidgeRegression:
        """
        Initialize a ridge regression model

        Parameters
        ----------
        lam: float
            Regularization parameter to be used when fitting a model

        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """


        """
        Initialize a ridge regression model
        :param lam: scalar value of regularization parameter
        """
        super().__init__()
        self.coefs_ = None
        self.include_intercept_ = include_intercept
        self.lam_ = lam

    def __add_ones(self, X: np.ndarray) -> np.ndarray:
        num_samples = len(X)
        if self.include_intercept_:
            X = np.c_[np.ones((num_samples, 1)), X]
        return X

    def get_X_y_lam(self, X, y = None):
        num_samples = len(X) + self.include_intercept_
        ident_mat = np.identity(num_samples)
        ident_mat_lam = ident_mat * np.sqrt(self.lam_)
        if (self.include_intercept_):
            X = np.c_[np.ones((num_samples, 1)), X]
            ident_mat_lam[0, 0] = 0
        if (self.lam_ != 0):
            X = X / ident_mat_lam
            # TODO: check value y is divided by (ident mat) - and check the shape of the division
            if (y != None):
                y = y/ident_mat
        return X, y
    


    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Ridge regression model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        #TODO: check if right
        X_lam, y_lam = self.get_X_y_lam(X, y)
        self.coefs_ = LinearRegression.fit(X_lam, y_lam)
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
        """
        X_lam, empty_val = self.get_X_y_lam(X)
        return X_lam @ self.coefs_

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        y_pred = self._predict(X)
        return loss_functions.mean_square_error(y, y_pred)
        raise NotImplementedError()