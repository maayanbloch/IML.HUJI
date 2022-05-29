from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from matplotlib import pyplot as plt

pio.templates.default = "simple_white"

FEATURES = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
            "waterfront", "view", "condition", "grade", "sqft_above",
            "sqft_basement", "yr_built", "yr_renovated", "zipcode",
            "sqft_living15", "sqft_lot15"]
PRICE = ["price"]
FEATURES_1 = ["bedrooms", "bathrooms", "sqft_living", "floors", "waterfront",
              "view", "grade", "sqft_above", "sqft_basement", "yr_renovated",
              "zipcode", "lat", "sqft_living15"]

REMOVE_FEATURES = ["price", "id", "date", "long", "condition", "sqft_lot",
                   "yr_built", "sqft_lot15"]
FEATURES_ALL = ["id", "date", "bedrooms", "bathrooms", "sqft_living",
                "sqft_lot", "floors", "waterfront", "view", "condition",
                "grade", "sqft_above", "sqft_basement", "yr_built",
                "yr_renovated", "zipcode", "lat", "long", "sqft_living15",
                "sqft_lot15"]
F = ["condition", "grade"]

MIN_PRICE = 40000
MAX_VIEW = 4
MAX_COND = 5
MAX_GRADE = 13
MAX_YEAR_BUILT = 2022


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    ##loading
    data_frame = pd.read_csv(filename)
    data_frame = data_frame.fillna(0)
    ##proccessing the date
    data_frame['date'] = pd.to_numeric(data_frame.date.str.replace('T', ""))

    ##proccessing the zipcode
    data_frame = data_frame.drop(data_frame[data_frame.zipcode <= 0].index)
    temp_zip = pd.get_dummies(data_frame['zipcode'])
    data_frame = data_frame.drop(['zipcode'], axis=1)
    data_frame = data_frame.join(temp_zip)

    ##proccessing the price
    data_frame = data_frame.drop(
        data_frame[data_frame.price < MIN_PRICE].index)

    ##proccessing the rest of the data to valid values
    data_frame = data_frame.drop(data_frame[data_frame.bathrooms < 0].index)
    data_frame = data_frame.drop(data_frame[data_frame.bedrooms < 0].index)
    data_frame = data_frame.drop(data_frame[data_frame.view < 0].index)
    data_frame = data_frame.drop(data_frame[data_frame.view > MAX_VIEW].index)
    data_frame = data_frame.drop(
        data_frame[data_frame.condition > MAX_COND].index)
    data_frame = data_frame.drop(data_frame[data_frame.condition < 1].index)
    data_frame = data_frame.drop(data_frame[data_frame.grade < 1].index)
    data_frame = data_frame.drop(
        data_frame[data_frame.grade > MAX_GRADE].index)
    data_frame = data_frame.drop(data_frame[data_frame.yr_built < 0].index)
    data_frame = data_frame.drop(
        data_frame[data_frame.yr_built > MAX_YEAR_BUILT].index)
    data_frame = data_frame.drop(data_frame[data_frame.yr_renovated < 0].index)
    data_frame = data_frame.drop(
        data_frame[data_frame.yr_renovated > MAX_YEAR_BUILT].index)
    data_frame = data_frame.drop(data_frame[(data_frame.waterfront != 0) & (
            data_frame.waterfront != 1)].index)

    ##remove duplicate rows
    data_frame = data_frame.drop_duplicates()

    ##remove rows with duplicate ids
    data_frame = data_frame.drop_duplicates(subset=['id'])

    X = data_frame.drop(REMOVE_FEATURES, axis=1)
    Y = data_frame.filter(items=PRICE)['price'].squeeze()
    return tuple((X, Y))


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    PC = X.apply(lambda column: y.cov(column))
    sig_x = X.std(axis=0)
    sig_y = y.std()
    PC = PC.divide((sig_x * sig_y))

    features = PC.index
    for i in range(12):
        fig = px.scatter(X, x=features[i], y=y,
                         title=(str(features[i]) + ": PC = " + str(PC[i])))
        filename = str(output_path + "/" + str(features[i]) + ".png")
        pio.write_html(fig, filename)


def calc_mean(sampled_train_por, train_X, y):
    temp_loss = np.zeros(10)
    for j in range(10):
        sampled_train_x = train_X.sample(frac=sampled_train_por)
        sampled_train_index = sampled_train_x.index
        sampled_train_y = y.loc[sampled_train_index]
        lin = LinearRegression()
        lin.fit(sampled_train_x, sampled_train_y)
        temp_loss[j] = lin.loss(test_X, test_y)
    return temp_loss.mean(), 2 * np.std(temp_loss)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("..\\datasets\\house_prices.csv")

    # Question 2 - Feature evaluation with respect to response

    feature_evaluation(X, y)

    # Question 3 - Split samples into training- and testing sets.
    train_proportion = 0.75
    train_X, train_y, test_X, test_y = split_train_test(X, y, train_proportion)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    n_samples = len(y)
    training_data_size = np.ceil(train_proportion * n_samples)
    mean_loss = np.zeros(90)
    std_loss = np.zeros(90)
    for i in range(10, 100):
        sampled_train_por = (0.01 * i)
        mean_loss[i - 10], std_loss[i - 10] = calc_mean(sampled_train_por,
                                                        train_X, y)
    perc = np.linspace(10, 100, 90)
    d = {'percentage': perc, 'mean loss': mean_loss, 'std loss': std_loss}
    mean_loss = pd.DataFrame(data=d)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=mean_loss['percentage'], y=mean_loss['mean loss'],
                   mode='lines', line_color='rgb(0,255,80)', showlegend=False))
    fig.add_traces([go.Scatter(x=mean_loss['percentage'],
                               y=mean_loss['mean loss'] + mean_loss[
                                   'std loss'], line_color='rgba(0,0,0,0)',
                               showlegend=False),
                    go.Scatter(x=mean_loss['percentage'],
                               y=mean_loss['mean loss'] - mean_loss[
                                   'std loss'], line_color='rgba(0,0,0,0)',
                               fill='tonexty', fillcolor='rgba(0,100,80,0.2)',
                               showlegend=False)])
    fig.update_layout(title='Percentage vs Mean Loss',
                      xaxis_title='Percentage', yaxis_title='Mean Loss')
    fig.show()
