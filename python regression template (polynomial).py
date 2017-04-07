#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 10:14:10 2017

@author: dana
"""

##polynomial regression template
##truth or bluff, predictions of real data values

##Polynomial Regression

##Importing the Libraries
import numpy as np ##for math stuffz
import matplotlib.pyplot as plt ##plot charts
import pandas as pd ##to import and manage datasets
import os as os ##to check working directory

##Set working directory
os.getcwd() 
os.chdir('/Users/Dana/Spyder/Gits/Machine-Learning-A-Z/Polynomial_Regression')

##Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
# use if you have more than 10 rows 
"""from sklearn.cross_vali dation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
"""

##fitting the regression model to the dataset
##create your regressor here

##predicting a new result 
y_pred = regressor.predict(6.5) ##linear regression predicts too high, polynomial predicts more accurate

##visualizing the regression results
x_grid = np.arange(min(x), max(x), 0.1) ##to create smooth/curved line
x_grid = x_grid.reshape((len(x_grid),1)) ##to create smooth/curved line

plt.scatter(x, y, color='red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()














