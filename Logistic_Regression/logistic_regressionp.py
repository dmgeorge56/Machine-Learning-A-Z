#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 12:22:03 2017

@author: dana
"""

##Logistic Regression 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os as os

##Set working directory
os.getcwd() 
os.chdir('/Users/Dana/Spyder/Gits/Machine-Learning-A-Z/Logistic_Regression')

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
sc_y = StandardScaler()
##y_train = sc_y.fit_transform(y_train) - do not scale, needs to stay 0 and 1

