#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 06:46:47 2017

@author: dana
"""

print ("hello world!")

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
dataset = pd.read_csv('Data.csv')

##distingiush independent variable matrix
x = dataset.iloc[:,:-1].values

##distinguish dependent variable matrix
y = dataset.iloc[:,3].values

##Missing Data
##replace missing cell w/ mean of column
from sklearn.preprocessing import Imputer ##for machine learning models/methods

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

##Categorical Variables
##encode variables
##make country dummies - place all categories of country into its own column of 0's and 1's
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0]) ##0=column
x = onehotencoder.fit_transform(x).toarray()

## ^: 3rd line fits the labelencoder_x object to the first column country
## ^: 3rd line returns matrix x encoded

##since the purchased column is the dependent variable, the machine learning program will know that
##it is a categorical variable
##so only need labelencoder, not onehotencoder
##so lets do purchased
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

##Splitting the data into training set and test set
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


##Feature Scaling
##age and income need to be on the same scale, because
##a lot of ML models are based on the euclidean distances 
##the squared root of the sum of the squared coordinates
##if the x coordinate is age, and y is salary, the euclidian distance will be
##dominated by salary
##so need to put the two on the same scale
##if not, the ML will basically be that the smaller values do not exist, and will 
##be dominated by the larger number
##two ways to feature scale:
    ##1. Standardization:
        #xstand = (x-meanx)/(sd(x)))
    ##2. Normalization:
        #xnorm = (x-minx)/(max(x)-min(x))
        
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler() #object
x_train= sc_x.fit_transform(x_train) #fit to object
x_test= sc_x.transform(x_test)


















