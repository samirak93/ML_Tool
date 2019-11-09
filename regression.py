from bokeh.palettes import RdBu

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

import pandas as pd
import numpy as np
import bisect

from math import pi
from numpy import arange
from itertools import chain
from collections import OrderedDict

def get_bounds(nlabels):
    bottom = list(chain.from_iterable([[ii]*nlabels for ii in range(nlabels)]))
    top = list(chain.from_iterable([[ii+1]*nlabels for ii in range(nlabels)]))
    left = list(chain.from_iterable([list(range(nlabels)) for ii in range(nlabels)]))
    right = list(chain.from_iterable([list(range(1,nlabels+1)) for ii in range(nlabels)]))
    return top, bottom, left, right

def get_colors(corr_array, colors):
    ccorr = arange(-1, 1, 1/(len(colors)/2))
    color = []
    for value in corr_array:
        ind = bisect.bisect_left(ccorr, value)
        color.append(colors[ind-1])
    return color


def get_corr_plot(df):
    corr = df.corr()
    colors = list(reversed(RdBu[9]))  # we want an odd number to ensure 0 correlation is a distinct color
    labels = df.columns
    nlabels = len(labels)
    top, bottom, left, right = get_bounds(nlabels)
    color_list = get_colors(corr.values.flatten(), colors)

    return top, bottom, left, right, labels, nlabels, color_list, corr.values.flatten()

def get_regression_plot(features_df, target_df):

    X_train, X_test, y_train, y_test = train_test_split(features_df, target_df, test_size=0.2, random_state=40)

    regressor = LinearRegression(normalize=True, n_jobs=-1)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    return y_test, y_pred