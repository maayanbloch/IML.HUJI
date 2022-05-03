from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, count = np.unique(y, return_counts=True)
        n_samples, n_features = X.shape
        self.pi_ = count / n_samples
        self.mu_ = []
        for c in self.classes_:
            self.mu_.append(np.mean(X[y == c], axis=0))
        self.mu_ = np.array(self.mu_)

        self.vars_ = np.zeros((len(self.classes_), n_features))
        for c_num, mu in enumerate(self.mu_):
            c = self.classes_[c_num]
            Si = np.zeros(n_features)
            for row in X[y == c]:
                t = (row - mu)
                Si += np.power(t,2)
            self.vars_[c_num] += (Si/(count[c_num]-1))
        self.fitted_ = True

    def __cov_and_inv(self):
        all_cov = np.eye(len(self.mu_[0])) * self.vars_[:, np.newaxis, :]
        all_cov_inv = np.linalg.inv(all_cov)
        return all_cov, all_cov_inv

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

        all_cov, all_cov_inv = self.__cov_and_inv()
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        likelihood = []
        for c in range(n_classes):
            cur_X = X - self.mu_[c]
            denom = np.sqrt(pow(2 * np.pi, n_features) * np.linalg.det(all_cov[c]))
            exp = (cur_X @ all_cov_inv[c] @ cur_X.T)
            like = (np.exp(-np.diag(exp) / 2) / denom) * self.pi_[c]
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