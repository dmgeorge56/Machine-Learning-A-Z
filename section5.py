#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 08:55:00 2017

@author: dana
"""

##Multiple Linear Regression

##Part 1: Data Preprocessing
##Importing the Libraries
import numpy as np ##for math stuffz
import matplotlib.pyplot as plt ##plot charts
import pandas as pd ##to import and manage datasets
import os as os ##to check working directory

##Set working directory
os.getcwd()
os.chdir('/Users/Dana/Spyder/Gits/Machine-Learning-A-Z')

##Importing the dataset
dataset = pd.read_csv('50_Startups.csv')

##distingiush independent variable matrix
x = dataset.iloc[:,:-1].values

##distinguish dependent variable matrix
y = dataset.iloc[:,4].values

##Categorical Variables
##encode variables
##make country dummies - place all categories of country into its own column of 0's and 1's
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3]) ##0=column
x = onehotencoder.fit_transform(x).toarray()

##avoiding dummy variable trap
x = x[:,1:]

## ^: 3rd line fits the labelencoder_x object to the first column country
## ^: 3rd line returns matrix x encoded

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

##fitting multiple linear regressino model to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

##predicting the test set results
y_pred = regressor.predict(x_test)












