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

    def get_X_lam(self, X):
        num_samples, n_features = X.shape
        ident_mat_lam = np.identity(n_features + self.include_intercept_) * np.sqrt(self.lam_)
        if (self.include_intercept_):
            X = np.c_[np.ones((num_samples, 1)), X]
            ident_mat_lam[0, 0] = 0
        #TODO: check that X_lam is right dims
        X_lam = np.stack((X, ident_mat_lam))
        return X_lam

    def get_y_lam(self, y, n_features):
        zeros_vec_d = np.zeros(n_features + self.include_intercept_)
        #TODO: check that y_lam is right dims
        y_lam = np.concatenate((y, zeros_vec_d), axis=None)
        return y_lam


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
        num_samples, n_features = X.shape
        X_lam = self.get_X_lam(X)
        y_lam = self.get_y_lam(y, n_features)
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
        # X_lam = self.get_X_lam(X)
        num_samples, n_features = X.shape
        if (self.include_intercept_):
            X = np.c_[np.ones((num_samples, 1)), X]
        return X @ self.coefs_

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
        #TODO: check loss function
        y_pred = self._predict(X)
        return loss_functions.mean_square_error(y, y_pred)