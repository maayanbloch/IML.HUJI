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

def plot_the_graphs(x, y, titles, full_title, mode_t, x_axis_index):
    fig = go.Figure()
    for i in range(len(y)):
        fig.add_trace(go.Scatter(x=x[x_axis_index[i]], y=y[i], mode=mode_t[i], name=titles[i]))
    fig.update_layout(
        title_text=full_title)
    fig.show()

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
    s = np.linspace(-1.2,2, num=n_samples, endpoint=True).reshape((n_samples, 1))
    s_index = np.arange(n_samples).reshape((n_samples, 1))
    sample_and_index_arr = np.hstack((s_index, s))
    sample_and_index = pd.DataFrame(sample_and_index_arr, columns=['inds', 'samples'])
    true_y = calc_f(s).flatten()
    noisy_y = true_y + np.random.normal(0, noise, n_samples)
    y_data_frame = pd.Series(noisy_y)
    train_p = 2/3
    train_X, train_y, test_X, test_y = split_train_test(sample_and_index, y_data_frame,
                                                        train_proportion=train_p)
    plot_the_graphs(x=[sample_and_index['samples'],train_X['samples'], test_X['samples']] , y=[true_y, train_y, test_y],
                    titles=["no noise y", "train y noise y", "test y noise y"], full_title=("noise = " + str(noise)), mode_t=["lines","markers", "markers"], x_axis_index=[0,1,2])


    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_err = np.empty(11)
    valid_err = np.empty(11)
    k_vec = np.arange(11)
    train_y_arr = pd.Series.to_numpy(train_y)
    train_x_arr_flat = train_X['samples'].to_numpy()
    train_x_arr = train_x_arr_flat.reshape((len(train_x_arr_flat), 1))
    test_y_arr = pd.Series.to_numpy(test_y)
    test_x_arr = test_X['samples'].to_numpy()
    for k in range(11):
        poly_model = PolynomialFitting(k)
        train_err[k], valid_err[k] = cross_validate(poly_model, train_x_arr, train_y_arr, mean_square_error, cv=5)

    plot_the_graphs(x=[k_vec],
                    y=[train_err, valid_err],
                    titles=["train error", "validation error"],
                    full_title=("noise = " + str(noise)),
                    mode_t=["markers", "markers"],
                    x_axis_index=[0, 0])

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = np.argmin(valid_err)
    k_poly_model = PolynomialFitting(best_k)
    k_poly_model.fit(train_x_arr_flat, train_y_arr)
    print("best k = " + str(best_k) + " noise = " + str(noise))
    test_err = k_poly_model.loss(test_x_arr, test_y_arr)
    print("test err = " + str(np.round(test_err, 2))  + " noise = " + str(noise))
    print("validation err = " + str(valid_err[best_k])  + " noise = " + str(noise))


def load_data(n_samples):
    X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
    train_p = n_samples/len(X)
    return split_train_test(X, y, train_proportion=train_p)


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
    ridge_lam_vals = np.linspace(0, 5, num=n_evaluations)
    lasso_lam_vals = np.linspace(0.1, 5.1, num=n_evaluations)

    ridge_train_loss = np.empty(n_evaluations)
    ridge_val_loss = np.empty(n_evaluations)
    lasso_train_loss = np.empty(n_evaluations)
    lasso_val_loss = np.empty(n_evaluations)
    train_X_arr = train_X.to_numpy()
    train_y_arr = train_y.to_numpy()
    for i in range(n_evaluations):
        ridge_model = RidgeRegression(lam=ridge_lam_vals[i])
        lasso_model = Lasso(alpha=lasso_lam_vals[i])
        ridge_train_loss[i], ridge_val_loss[i] = cross_validate(ridge_model, train_X_arr, train_y_arr, mean_square_error, cv=5)
        lasso_train_loss[i], lasso_val_loss[i] = cross_validate(lasso_model, train_X_arr, train_y_arr, mean_square_error ,cv=5)

    plot_the_graphs(x=[ridge_lam_vals], y=np.array([ridge_train_loss, ridge_val_loss]),
                    titles=["train error", "validation error"],
                    full_title="Ridge train and validation err", mode_t=["markers", "markers"], x_axis_index=[0,0])

    plot_the_graphs(x=[lasso_lam_vals],
                    y=np.array([lasso_train_loss, lasso_val_loss]),
                    titles=["train error", "validation error"],
                    full_title="Lasso train and validation err",
                    mode_t=["markers", "markers"],x_axis_index=[0,0])



    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_lam_ridge = ridge_lam_vals[np.argmin(ridge_val_loss)]
    best_lam_lasso = lasso_lam_vals[np.argmin(lasso_val_loss)]
    best_ridge = RidgeRegression(lam=best_lam_ridge)
    best_lasso = Lasso(alpha=best_lam_lasso)
    best_ridge.fit(train_X, train_y)
    best_lasso.fit(train_X, train_y)
    #TODO: find least square loss i implemented
    ridge_test_error = best_ridge.loss(test_X, test_y)
    lasso_best_y = best_lasso.predict(test_X)
    lasso_test_error = mean_square_error(lasso_best_y, test_y)
    print("Ridge test error = " + str(
        ridge_test_error) + " for lamda value " + str(best_lam_ridge))
    print("Lasso test error = " + str(
        lasso_test_error) + " for lamda value " + str(best_lam_lasso))


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
