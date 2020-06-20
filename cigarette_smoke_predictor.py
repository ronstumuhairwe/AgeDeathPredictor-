# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 16:52:45 2020
@author:Ronald Tumuhairwe
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv('weed_smoking_dataset.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=1/2, random_state=0)

#Fitting the training set onto the regression model
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

y_pred = regressor.predict(X_test)

plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('WeedSmoked vs AgeDeath(Training Set)')
plt.xlabel('Quantity of weed Smoke (sticks)')
plt.ylabel('Age Death (years)')
plt.show()








