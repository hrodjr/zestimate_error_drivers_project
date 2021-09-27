import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, QuantileTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from math import sqrt


#gets the baseline
def get_baseline(df, x, y):
    df['yhat_baseline'] = y.mean()
    model = LassoLars().fit(x, y)
    df['yhat'] = model.predict(x)
    return df

#gets residuals
def get_residuals(df, y):
    df['residual'] = df['yhat'] - y
    df['residual_baseline'] = df['yhat_baseline'] - y
    return df

#creates a residual plot
def plot_residual(df, x, y):
    sns.residplot(x, y, color = 'orange')
    plt.show()

#returns regression errors
def regression_errors(df, y, yhat):
    MSE2 = mean_squared_error(y, yhat)
    SSE2 = MSE2 * len(df)
    ESS = sum((yhat - y.mean())**2)
    TSS = ESS + SSE2
    RMSE2 = mean_squared_error(y, yhat, squared = False)
    return (MSE2, SSE2, ESS, TSS, RMSE2)

#returns regression errors
def regression_errors(df, y, yhat):
    MSE2 = mean_squared_error(y, yhat)
    SSE2 = MSE2 * len(df)
    ESS = sum((yhat - y.mean())**2)
    TSS = ESS + SSE2
    RMSE2 = mean_squared_error(y, yhat, squared = False)
    return (MSE2, SSE2, ESS, TSS, RMSE2)

#computes the SSE, MSE, and RMSE for the baseline model
def baseline_mean_errors(df, y, yhat_baseline):
    MSE2_baseline = mean_squared_error(y, yhat_baseline)
    SSE2_baseline = MSE2_baseline * len(df)
    RMSE2_baseline = mean_squared_error(y, yhat_baseline, squared=False)
    
    return MSE2_baseline, SSE2_baseline, RMSE2_baseline

#returns true if your model performs better than the baseline, otherwise false
def better_than_baseline(regression_errors = True, baseline_mean_errors = True):
    
    if regression_errors - baseline_mean_errors <  regression_errors:
        print('The model is better then the baseline.')
    else:
        print('The model is not better then the baseline.')

def rfe(X_train,y_train,k):
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=k)
    rfe.fit(X_train_scaled, y_train)
    return X_train.columns[rfe.get_support()]

def select_kbest(X_train,y_train,k):
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    kbest = SelectKBest(f_regression, k=k)
    kbest.fit(X_train_scaled, y_train)
    features = kbest.get_support()
    return X_train.columns[kbest.get_support()]

def scale_it(X_train, X_validate, X_test):
    scaler = MinMaxScaler()
# Note that we only call .fit with the training data,
# but we use .transform to apply the scaling to all the data splits.
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)

    plt.figure(figsize=(13, 6))
    plt.subplot(121)
    plt.hist(X_train, bins=15, ec='black')
    plt.title('Original')
    plt.subplot(122)
    plt.hist(X_train_scaled, bins=15, ec='black')
    plt.title('Scaled')