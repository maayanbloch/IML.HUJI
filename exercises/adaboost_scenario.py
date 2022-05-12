import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size
    Parameters
    ----------
    n: int
        Number of samples to generate
    noise_ratio: float
        Ratio of labels to invert
    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples
    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y

def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    wl = lambda:DecisionStump()
    ada = AdaBoost(wl, n_learners).fit(train_X, train_y)
    train_err = []
    test_err = []
    for i in range(n_learners):
        train_err.append(ada.partial_loss(train_X, train_y, i+1))
        test_err.append(ada.partial_loss(test_X, test_y, i+1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_err, mode="markers+lines", name="train error"))
    fig.add_trace(go.Scatter(y=test_err, mode="markers+lines", name="test error"))
    fig.update_layout(
        title_text="noise = " + str(noise))
    fig.show()


    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    symbols = np.array(["circle", "square", "triangle-up"])
    y_true = np.array(test_y, dtype=np.int)
    for t in T:
        def pred(X):
            return ada.partial_predict(X, t)

        fig = go.Figure()
        fig.add_traces([decision_surface(pred, lims[0], lims[1],
                                        showscale=False), go.Scatter(x=test_X[:,0], y=test_X[:,1]
                                     , mode="markers", showlegend=False,
                    marker=dict(color=y_true, symbol=symbols[y_true], colorscale=[custom[0], custom[-1]],
                                line=dict(color="black", width=1)))])
        fig.update_layout(
            title_text="T is = " + str(t) + " noise = " + str(noise))
        fig.show()


    # Question 3: Decision surface of best performing ensemble
    lowest_error_size = np.min(np.argmin(np.array(test_err))) + 1

    def pred(X):
        return ada.partial_predict(X, lowest_error_size)
    lowest_accuracy = 1 - test_err[lowest_error_size - 1]
    fig = go.Figure()
    y_true = np.array(test_y, dtype=np.int)
    fig.add_traces(
        [decision_surface(pred, lims[0], lims[1],
                          showscale=False),
         go.Scatter(x=test_X[:, 0], y=test_X[:, 1]
                    , mode="markers", showlegend=False,
                    marker=dict(color=y_true, symbol=symbols[y_true],
                                colorscale=[custom[0], custom[-1]],
                                line=dict(color="black", width=1)))])
    fig.update_layout(
        title_text="T is = " + str(lowest_error_size) + " accuracy = " + str(lowest_accuracy) + " noise = " + str(noise))
    fig.show()


    # Question 4: Decision surface with weighted samples
    D = 5* ada.D_/np.max(ada.D_)
    fig = go.Figure()
    y_true = np.array(train_y, dtype=np.int)
    fig.add_traces(
        [decision_surface(ada.predict, lims[0], lims[1],
                          showscale=False),
         go.Scatter(x=train_X[:, 0], y=train_X[:, 1]
                    , mode="markers", showlegend=False,
                    marker=dict(size=D,color=y_true, symbol=symbols[y_true],
                                colorscale=[custom[0], custom[-1]],
                                line=dict(color="black", width=1)))])
    fig.update_layout(
        title_text="T is = " + str(n_learners) + " noise = " + str(noise))
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)



