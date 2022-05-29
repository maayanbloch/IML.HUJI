import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"

COUNTRIES = ["South Africa", "The Netherlands", "Israel", "Jordan"]
NO_ISR = ["South Africa", "The Netherlands", "Jordan"]
def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data_frame = pd.read_csv(filename, parse_dates=['Date'])
    data_frame = data_frame.loc[data_frame['Country'].isin(COUNTRIES)]
    data_frame = data_frame.loc[data_frame['Month'].isin(np.arange(start=1, stop=13, step=1, dtype=int))]
    data_frame = data_frame.drop(data_frame[data_frame.Temp < -60].index)
    data_frame['dayofyear'] = data_frame['Date'].dt.dayofyear
    data_frame = data_frame.fillna(0)
    data_frame = data_frame.drop_duplicates()
    return data_frame





if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    X= load_data("..\\datasets\\City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    Israel_data = X.drop(X[X.Country != "Israel"].index)
    Israel_data['Year'] = Israel_data['Year'].apply(str)
    fig = px.scatter(x=Israel_data['dayofyear'], y=Israel_data['Temp'], color=Israel_data['Year'], title='day of year vs temp (by year)')
    fig.show()

    Israel_g = Israel_data.groupby('Month')['Temp'].agg([np.std])
    fig = px.bar(Israel_g, x =Israel_g.index, y='std')
    fig.show()


    # Question 3 - Exploring differences between countries
    C_M_group= X.groupby(['Country', 'Month'], as_index=True).agg({'Temp':[np.mean, np.std]}).reset_index()
    C_M_group.columns = [' '.join(t).strip() for t in C_M_group.columns]
    fig = px.line(C_M_group, x='Month', y='Temp mean', color='Country', error_y='Temp std', error_y_minus='Temp std')
    fig.show()

    # Question 4 - Fitting model for different values of `k`
    Israel_dayofyear = Israel_data.filter(items = ['dayofyear']).squeeze()
    Israel_temp = Israel_data.filter(items=['Temp']).squeeze()
    train_proportion = 0.75
    loss = np.zeros(10)
    err = np.zeros(10)
    train_X, train_y, test_X, test_y = split_train_test(Israel_dayofyear, Israel_temp, train_proportion)
    for k in range(1, 11):
        pol = PolynomialFitting(k)
        pol.fit(train_X, train_y)
        loss[k-1] = round(pol.loss(test_X, test_y), 2)
    loss = pd.DataFrame(loss, columns=['loss'])
    loss.index = loss.index+1
    print(loss)
    fig = px.bar(loss,x =loss.index , y='loss')
    fig.show()

    # Question 5 - Evaluating fitted model on different countries
    k = 5
    pol = PolynomialFitting(k)
    X_day = X.filter(items=['dayofyear', 'Country'])
    y_day = X.filter(items=['Temp', 'Country'])
    pol.fit(Israel_dayofyear, Israel_temp)
    c_loss = np.zeros(3)
    for i in range(3):
        cur_X = X.drop(X[X.Country != NO_ISR[i]].index)
        c_x_data = cur_X.filter(items =['dayofyear']).squeeze()
        c_y_data = cur_X.filter(items=['Temp']).squeeze()
        c_loss[i] = pol.loss(c_x_data, c_y_data)
    c_loss = pd.DataFrame(c_loss, columns=['loss'])
    c_loss.index = NO_ISR
    fig = px.bar(c_loss, x=c_loss.index, y='loss')
    fig.show()