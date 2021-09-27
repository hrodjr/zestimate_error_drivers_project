import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

#plots train variable distributions
def variable_distributions(train):
 
    for col in train.columns:
        plt.figure(figsize=(4,2))
        plt.ticklabel_format(axis = 'both', style = 'sci', useMathText = True)
        plt.hist(train[col], edgecolor='black')
        plt.title(col)
        plt.show()

#plots variables
def plot_against_target(train, target, var_list, figsize = (10,5), hue = None):   
    '''
    Takes in dataframe, target and varialbe list, and plots against target. 
    '''
    for var in var_list:
        plt.figure(figsize = (figsize))
        plt.ticklabel_format(axis = 'both', style = 'sci', useMathText = True)
        sns.regplot(data = train, x = var, y = target, color = 'orange', line_kws={'color': 'green'})
        plt.xlabel(var)
        plt.ylabel('Log Error')
        plt.show()

def pairplot_distribution (train):
#columns
    cols = ['age', 'latitude', 'longitude', 'logerror']

    sns.pairplot(data = [cols], corner=True)
    plt.suptitle('Amount of error is to see with Logerror', fontsize = 15)
    plt.show()

def inertia(X):
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(9, 6))
        pd.Series({k: KMeans(k).fit(X).inertia_ for k in range(2, 12)}).plot(marker='x')
        plt.xticks(range(2, 12))
        plt.xlabel('k')
        plt.ylabel('inertia')
        plt.title('Change in inertia as k increases')