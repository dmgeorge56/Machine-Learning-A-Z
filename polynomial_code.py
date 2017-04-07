#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 18:53:18 2017

@author: dana
"""

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
##fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

##fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4) ##or 2 for squared, pick by the fit of the model on the plot
x_poly = poly_reg.fit_transform(x)

##new linear regression object to use poly_reg object in linear reg model
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

##visualizing the linear regression results 
plt.scatter(x,y, color='red')
plt.plot(x,lin_reg.predict(x), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show

##visualizing the polynomial regression results
x_grid = np.arange(min(x), max(x), 0.1) ##to create curved line
x_grid = x_grid.reshape((len(x_grid),1)) ##to create curved line

plt.scatter(x, y, color='red')
plt.plot(x_grid, lin_reg2.predict(poly_reg.fit_transform(x_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show

##predicting a new result with linear regression
lin_reg.predict(6.5) ##predict salary of new employee

##predicting a new result with polynomial regression
lin_reg2.predict(poly_reg.fit_transform(6.5))

##linear regression predicts too high, polynomial predicts more accurate















