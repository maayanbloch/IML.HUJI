from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ , count = np.unique(y, return_counts=True)
        n_samples, n_features = X.shape
        self.pi_ = count / n_samples
        self.mu_ = []
        for c in self.classes_:
            self.mu_.append(np.mean(X[y == c], axis=0))
        self.mu_ = np.array(self.mu_)
        self.cov_ = np.zeros((n_features, n_features))
        for c_num, mu in enumerate(self.mu_):
            c = self.classes_[c_num]
            Si = np.zeros((n_features, n_features))
            for row in X[y==c]:
                t = (row-mu).reshape(1, n_features)
                Si += np.dot(t.T, t)
            self.cov_ += Si
        self.cov_ = self.cov_ / (n_samples - n_features)
        self._cov_inv = inv(self.cov_)
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
        likelihood = self.likelihood(X)
        res = np.argmax(likelihood, axis=1)

        return self.classes_[res]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        cov_det = det(self.cov_)
        likelihood = []
        for c in range(n_classes):
            cur_X = X - self.mu_[c]
            denom = np.sqrt(pow(2 * np.pi, n_features) * cov_det)
            exp = (cur_X @ self._cov_inv @ cur_X.T)
            like = (np.exp(-np.diag(exp)/2)/denom)*self.pi_[c]
            likelihood.append(like)
        return np.array(likelihood).T


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
        from ...metrics import misclassification_error
        y_pred = self._predict(X)
        return misclassification_error(y, y_pred, normalize=False)