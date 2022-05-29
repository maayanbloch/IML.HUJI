from __future__ import annotations
import numpy as np
import pandas as pd
# from pandas.io import pickle
import pickle
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def calc_f(x):
    return (x+3)*(x+2)*(x+1)*(x-1)*(x-2)

def model_function(train_ind, test_ind, X):
    train_y = calc_f(X[train_ind])
    test_y = calc_f(X[test_ind])
    return train_y, test_y



def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    s = np.linspace(-1.2,2, num=n_samples, endpoint=True)
    s = s.reshape((n_samples, 1))
    shuffled_indexes = np.arange(n_samples)
    np.random.shuffle(shuffled_indexes)
    #TODO: use my implementation of split to test in utils
    train_ind, test_ind = np.split(shuffled_indexes, [int(np.round(n_samples*2/3))])
    true_y = calc_f(s).flatten()
    noise_vec = np.random.normal(0, noise, n_samples)
    noisy_y = true_y + noise_vec
    color_array = np.ones(n_samples)
    color_array[train_ind] = 0
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(y=true_y, mode="lines", name="no noise y"))
    fig.add_trace(
        go.Scatter(y=noisy_y, mode="markers", name="noise y",  marker=dict(color=color_array)))
    fig.update_layout(
        title_text="noise = " + str(noise))
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_err = np.empty(11)
    valid_err = np.empty(11)
    k_vec = np.arange(10)
    for k in range(11):
        poly_model = PolynomialFitting(k)
        train_err[k], valid_err[k] = cross_validate(poly_model, s[train_ind], noisy_y[train_ind], mean_square_error, cv=5)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(y=train_err, mode="markers+lines", name="train error",
                   marker=dict(color=k_vec)))
    fig.update_layout(
        title_text="train err, noise = " + str(noise))
    fig.show()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(y=valid_err, mode="markers+lines", name="validation error",
                   marker=dict(color=k_vec)))
    fig.update_layout(
        title_text="validation err, noise = " + str(noise))
    fig.show()
    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = np.argmin(valid_err)
    k_poly_model = PolynomialFitting(best_k)
    k_poly_model.fit(s[train_ind].flatten(), noisy_y[train_ind].flatten())
    print("best k = " + str(best_k) + " noise = " + str(noise))
    test_err = k_poly_model.loss(s[test_ind].flatten(), noisy_y[test_ind].flatten())
    print("test err = " + str(np.round(test_err, 2))  + " noise = " + str(noise))
    print("validation err = " + str(valid_err[best_k])  + " noise = " + str(noise))


def load_data(n_samples):
    data_frame = pickle.load(open("./diabetes", "rb"))
    X = data_frame.drop(y, axis=1)
    y = data_frame.filter(items=y)['y'].squeeze()
    train_p = 50/len(X)
    #TODO: get split train test function
    train_X, train_y, test_X, test_y = split_train_test(data_frame, y, train_proportion=train_p)
    return train_X, train_y, test_X, test_y

def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    train_X, train_y, test_X, test_y = load_data(n_samples)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    #TODO: find relevant lam values
    ridge_lam_vals = np.linspace(0, 400, num=n_evaluations)
    lasso_lam_vals = np.linspace(0, 400, num=n_evaluations)

    ridge_train_loss = np.empty(n_evaluations)
    ridge_val_loss = np.empty(n_evaluations)
    lasso_train_loss = np.empty(n_evaluations)
    lasso_val_loss = np.empty(n_evaluations)

    for i in range(n_evaluations):
        ridge_model = RidgeRegression(lam=ridge_lam_vals[i])
        lasso_model = Lasso(alpha=lasso_lam_vals[i])
        ridge_train_loss[i], ridge_val_loss[i] = cross_validate(ridge_model, train_X, train_y, mean_square_error, cv=5)
        lasso_train_loss[i], lasso_val_loss[i] = cross_validate(lasso_model, train_X, train_y, mean_square_error ,cv=5)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=ridge_lam_vals, y=ridge_train_loss, mode="markers+lines", name="train error"))
    fig.add_trace(
        go.Scatter(x=ridge_lam_vals, y=ridge_val_loss, mode="markers+lines",
                   name="validation error"))
    fig.update_layout(
        title_text="Ridge train and validation err")
    fig.show()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=lasso_lam_vals, y=lasso_train_loss, mode="markers+lines",
                   name="train error"))
    fig.add_trace(
        go.Scatter(x=lasso_lam_vals, y=lasso_val_loss, mode="markers+lines",
                   name="validation error"))
    fig.update_layout(
        title_text="Lasso train and validation err")
    fig.show()



    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_lam_ridge = ridge_lam_vals[np.argmin(ridge_val_loss)]
    best_lam_lasso = lasso_lam_vals[np.argmin(lasso_val_loss)]
    best_ridge = RidgeRegression(lam=best_lam_ridge)
    best_lasso = Lasso(alpha=best_lam_lasso)
    best_ridge.fit(train_X, train_y)
    best_lasso.fit(train_X, train_y)
    #TODO: find least square loss i implemented
    ridge_test_error = best_ridge.loss(test_X, test_y)
    lasso_test_error = best_lasso.loss(test_X, test_y)
    print("Ridge test error = " + str(
        ridge_test_error) + " for lamda value " + str(best_lam_ridge))
    print("Lasso test error = " + str(
        lasso_test_error) + " for lamda value " + str(best_lam_lasso))


if __name__ == '__main__':
    np.random.seed(0)
    # select_polynomial_degree()
    # select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    raise NotImplementedError()
