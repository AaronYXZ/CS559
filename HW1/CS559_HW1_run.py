## Change directory



## import Packages
import CS559_HW1_setup as ch 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

## Get the data
train = pd.read_csv('hw1_train.csv', index_col = 0)
test = pd.read_csv('hw1_test.csv', index_col = 0)

X_train = train.ix[:,:-1].values
y_train = train.ix[:,-1].values
X_test = test.ix[:,:-1].values
y_test = test.ix[:,-1].values


## Build a model object from self-defined package
model = ch.Logistic_Regression(0.01, 1, 200)

## Fit the model on training data, return prediction accuracy on testing data
t = model.score(X_train,y_train, X_test, y_test, 0.5)
print(t)



