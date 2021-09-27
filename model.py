import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

def model_baseline(y_train, y_validate, target):
 #Convert y_train and y_validate to be dataframes to append the new columns with predicted values.    
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)

#1. Predict logerror_pred_mean
    logerror_pred_mean = y_train[target].mean()
    y_train['logerror_pred_mean'] = logerror_pred_mean
    y_validate['logerror_pred_mean'] = logerror_pred_mean

#2. compute logerror_pred_median
    logerror_pred_median = y_train[target].median()
    y_train['logerror_pred_median'] = logerror_pred_median
    y_validate['logerror_pred_median'] = logerror_pred_median

#3. RMSE of logerror_pred_mean
    rmse_train = mean_squared_error(y_train.logerror, y_train.logerror_pred_mean)**(1/2)
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.logerror_pred_mean)**(1/2)
    print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))

#4. RMSE of logerror_pred_median
    rmse_train = mean_squared_error(y_train.logerror, y_train.logerror_pred_median)**(1/2)
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.logerror_pred_median)**(1/2)
    print("RMSE using Median\nTrain/In-Sample: ", round(rmse_train, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))

def linear_regression(y_train, X_train, y_validate, X_validate):
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
# create the model object
    lm = LinearRegression(normalize=True)

# fit the model to our training data. We must specify the column in y_train, 
# since we have converted it to a dataframe from a series! 
    lm.fit(X_train, y_train['logerror'])

# predict train
    y_train['logerror_pred_lm'] = lm.predict(X_train)

# evaluate: rmse
    rmse_train = mean_squared_error(y_train.logerror, y_train.logerror_pred_lm)**(1/2)

# predict validate
    y_validate['logerror_pred_lm'] = lm.predict(X_validate)

# evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.logerror_pred_lm)**(1/2)

    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate)

def lassolars(y_train, X_train, y_validate, X_validate):
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
# create the model object
    lars = LassoLars(alpha=1.0)

# fit the model to our training data. We must specify the column in y_train, 
# since we have converted it to a dataframe from a series! 
    lars.fit(X_train, y_train.logerror)

# predict train
    y_train['logerror_pred_lars'] = lars.predict(X_train)

# evaluate: rmse
    rmse_train = mean_squared_error(y_train.logerror, y_train.logerror_pred_lars)**(1/2)

# predict validate
    y_validate['logerror_pred_lars'] = lars.predict(X_validate)

# evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.logerror_pred_lars)**(1/2)

    print("RMSE for Lasso + Lars\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate)

def tweedieregressor(y_train, X_train, y_validate, X_validate):
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
# create the model object
    glm = TweedieRegressor(power=1, alpha=0)

# fit the model to our training data. We must specify the column in y_train, 
# since we have converted it to a dataframe from a series! 
    glm.fit(X_train, y_train.logerror)

# predict train
    y_train['logerror_pred_glm'] = glm.predict(X_train)

# evaluate: rmse
    rmse_train = mean_squared_error(y_train.logerror, y_train.logerror_pred_glm)**(1/2)

# predict validate
    y_validate['logerror_pred_glm'] = glm.predict(X_validate)

# evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.logerror_pred_glm)**(1/2)

    print("RMSE for GLM using Tweedie, power=1 & alpha=0\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate) 

def polynomialregression(y_train, X_train, y_validate, X_validate, X_test):
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
# make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)

# fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train)

# transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(X_validate)
    X_test_degree2 = pf.transform(X_test)

# create the model object
    lm2 = LinearRegression(normalize=True)

# fit the model to our training data. We must specify the column in y_train, 
# since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, y_train.logerror)

# predict train
    y_train['logerror_pred_lm2'] = lm2.predict(X_train_degree2)

# evaluate: rmse
    rmse_train = mean_squared_error(y_train.logerror, y_train.logerror_pred_lm2)**(1/2)

# predict validate
    y_validate['logerror_pred_lm2'] = lm2.predict(X_validate_degree2)

# evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.logerror_pred_lm2)**(1/2)

    print("RMSE for Polynomial Model, degrees=2\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate)

def model_test(X_test, y_test):
    X_test = pd.DataFrame(X_test)
    y_test = pd.DataFrame(y_test)
    pf = PolynomialFeatures(degree=2)
    X_test_degree2 = pf.fit_transform(X_test)
    X_test_degree2 = pf.transform(X_test)
    lm2 = LinearRegression(normalize=True) 
    lm2.fit(X_test_degree2, y_test.logerror)
    y_test['logerror_pred_lm2'] = lm2.predict(X_test_degree2)
    rmse_test = mean_squared_error(y_test.logerror, y_test.logerror_pred_lm2)**(1/2)
    print("RMSE for Polynomial Model, degrees=2\nTest: ", rmse_test)
    
def linear_regression_test(X_test, y_test):
    X_test = pd.DataFrame(X_test)
    y_test = pd.DataFrame(y_test)
    lm = LinearRegression(normalize=True)
    lm.fit(X_test, y_test['logerror'])
    y_test['logerror_pred_lm'] = lm.predict(X_test)
    rmse_test = mean_squared_error(y_test.logerror, y_test.logerror_pred_lm)**(1/2)
    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_test)

def lassolars_test(X_test, y_test):
    X_test = pd.DataFrame(X_test)
    y_test = pd.DataFrame(y_test)
    lars = LassoLars(alpha=1.0)
    lars.fit(X_test, y_test.logerror)
    y_test['logerror_pred_lars'] = lars.predict(X_test)
    rmse_test = mean_squared_error(y_test.logerror, y_test.logerror_pred_lars)**(1/2)
    print("RMSE for Lasso + Lars\nTraining/In-Sample: ", rmse_test)


def plot_test_residuals():
    plt.figure(figsize=(16,8))
    plt.scatter(y_test.logerror, y_test.logerror_pred_lm2, alpha=.5, color="purple", s=100)
    plt.xlabel("Actual logerror")
    plt.ylabel("Predicted logerror")
    plt.title("Residuals")
    plt.show()

#plots county code tax rate distributions
def plot_distributions(counties, x_value, group_key):
    
    for county in counties:
        sns.distplot(county[x_value])
        plt.title(county[group_key].unique())
        plt.show()
        print()
        print(county[x_value].describe())