from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
import pandas as pd


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """

    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        filename = "../datasets/" + f
        X, y_true = load_dataset(filename)

        # Fit Perceptron and record loss in each fit iteration
        iter_loss = []
        def callback_func(fit: Perceptron, x: np.ndarray, y: int):
            iter_loss.append(fit.loss(X, y_true))

        Perceptron(callback=callback_func).fit(X, y_true)

        # Plot figure of loss as function of fitting iteration
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(y=iter_loss,
                       mode='lines', line_color='rgb(0,255,80)',
                       showlegend=False))
        fig.update_layout(title=n + " vs iteration number",
                          xaxis_title='iteration number', yaxis_title=n)
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        filename = "../datasets/" + f
        X, y_true = load_dataset(filename)

        # Fit models and predict over training set
        LDA_inst = LDA().fit(X, y_true)
        LDA_y_pred = LDA_inst.predict(X)
        GNA_inst = GaussianNaiveBayes().fit(X, y_true)
        GNA_y_pred = GNA_inst.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        LDA_acc = accuracy(y_true, LDA_y_pred)
        GNA_acc = accuracy(y_true, GNA_y_pred)
        LDA_title = "LDA algorithm, accuracy = " + str(LDA_acc)
        GNA_title = "GNA algorithm, accuracy = " + str(GNA_acc)
        fig = make_subplots(rows=1, cols=2, subplot_titles= (GNA_title, LDA_title))

        # Add traces for data-points setting symbols and colors
        symbols = np.array(["circle", "square", "triangle-up"])
        y_true = np.array(y_true, dtype=np.int)
        fig.add_trace(
            go.Scatter(mode='markers', x=X.T[0], y=X.T[1],
                       marker=dict(color=GNA_y_pred, symbol=symbols[y_true])),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(mode='markers', x=X.T[0],y=X.T[1],
                       marker=dict(color=LDA_y_pred, symbol=symbols[y_true])),
            row=1, col=2
        )

        fig.update_layout(
            title_text="dataset used = " + f)


        # Add `X` dots specifying fitted Gaussians' means
        fig.add_trace(
            go.Scatter(mode='markers',x = GNA_inst.mu_.T[0] ,y=GNA_inst.mu_.T[1],
                       marker=dict(color="black", symbol='x')),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(mode='markers', x = LDA_inst.mu_.T[0] ,y=LDA_inst.mu_.T[1],
                       marker=dict(color="black", symbol='x')),
            row=1, col=2
        )


        # Add ellipses depicting the covariances of the fitted Gaussians
        classes = LDA_inst.classes_
        for c in range(len(classes)):
            GNA_cov = np.eye(len(X[0])) * GNA_inst.vars_[c]
            fig.add_trace(get_ellipse(GNA_inst.mu_[c],GNA_cov), row=1, col=1)
            fig.add_trace(get_ellipse(LDA_inst.mu_[c],LDA_inst.cov_), row=1, col=2)


        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
