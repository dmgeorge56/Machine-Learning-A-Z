#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 15:27:40 2017

@author: dana
"""

##Random Forests Intuition Notes
##applied to regression trees
##ensemble learning:
    ##when you take multiple algorithms and put 
    ##them together to make something more powerful
    ##than the original
##get multiple predictions and take the average
##of the predictions over all of the trees
##reminds him of:
        ##take a guess how many m&ms are in the jar
        ##best way to beat that game is
        ##get a pen and paper and stand next to
        ##the person that takes the guesses
        ##take the average of all the guesses
        ##by everyone else
        ##you have more statistical power doing this
        ##than by guessing alone
##Random forest = a team of decision trees

##Importing the Libraries
import numpy as np ##for math stuffz
import matplotlib.pyplot as plt ##plot charts
import pandas as pd ##to import and manage datasets
import os as os ##to check working directory

##Set working directory
os.getcwd() 
os.chdir('/Users/Dana/Spyder/Gits/Machine-Learning-A-Z/Random_Forest_Regression')

##Importing the dataset 
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(x, y)

# Predicting a new result
y_pred = regressor.predict(6.5)

# Visualising the Random Forest Regression results (higher resolution)
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show() 