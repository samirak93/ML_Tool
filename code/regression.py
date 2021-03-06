from bokeh.palettes import RdBu

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer, OneHotEncoder, LabelEncoder
from sklearn import metrics

import pandas as pd
import numpy as np
import bisect

from numpy import arange
from itertools import chain
from collections import defaultdict

def get_bounds(nlabels):
    bottom = list(chain.from_iterable([[ii]*nlabels for ii in range(nlabels)]))
    top = list(chain.from_iterable([[ii+1]*nlabels for ii in range(nlabels)]))
    left = list(chain.from_iterable([list(range(nlabels)) for ii in range(nlabels)]))
    right = list(chain.from_iterable([list(range(1,nlabels+1)) for ii in range(nlabels)]))
    return top, bottom, left, right

def get_colors(corr_array, colors, min, max):
    ccorr = arange(min, max, 1/(len(colors)/2))
    color = []
    for value in corr_array:
        ind = bisect.bisect_left(ccorr, value)
        color.append(colors[ind-1])

    return color

def get_corr_plot(df):

    corr = df.corr()
    colors = list(reversed(RdBu[9]))
    labels = df.columns
    nlabels = len(labels)
    top, bottom, left, right = get_bounds(nlabels)
    color_list = get_colors(corr.values.flatten(), colors, -1, 1)

    return top, bottom, left, right, labels, nlabels, color_list, corr.values.flatten()

def get_regression_plot(features_df, target_df, active_norm):

    non_num_features = [col for col, dt in features_df.dtypes.items() if dt == object]
    likely_cat = {}
    for var in features_df.columns:
                likely_cat[var] = features_df[var].nunique() <= 100
    likely_cat = [k for k, v in likely_cat.items() if v is True]
    non_num_features = list(set(non_num_features + likely_cat))

    if list(non_num_features):
        lb_results_df = pd.DataFrame(pd.get_dummies(features_df[non_num_features]))
        features_df = features_df.drop(columns=non_num_features)
        features_df = pd.concat([features_df, lb_results_df], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(features_df, target_df, test_size=0.2, random_state=40)

    if active_norm == 1:
        X_train = StandardScaler().fit_transform(X_train)
        X_test = StandardScaler().fit_transform(X_test)
    else:
        X_train = X_train
        X_test = X_test

    regressor = LinearRegression(normalize=True, n_jobs=-1)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)

    residual = y_test - y_pred

    r2 = metrics.r2_score(y_test, y_pred)
    slope = regressor.coef_[0]
    intercept = regressor.intercept_

    text = ["R^2 - %02f" % r2]
    MAE = np.round(metrics.mean_absolute_error(y_test, y_pred),2)
    MSE = np.round(metrics.mean_squared_error(y_test, y_pred),2)
    RMSE = np.round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2)

    return y_test, y_pred, text, MAE, RMSE, residual, slope, intercept