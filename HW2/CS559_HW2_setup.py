
import numpy as np 
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, log_loss

class Neural_Network(object):
    def __init__(self, hidden_units = 3, step_size = 0.001, iteration = 1000, regualization = 0.1):
        self.units = hidden_units
        self.ss = step_size
        self.ite = iteration
        self.reg = regualization

    def makeData(self, start, end, size = 600, dimension = 2):
        X = np.zeros((size, dimension))
        for i in range(dimension):
            X[:,i] = np.random.uniform(start, end, size = size)
        y = np.array([sum( [int(i) for i in record]) % 2 for record in X]) 

        return X, y

    def relu(self, a):
        ## Computes first layer using the ReLU function
        return a * (a>0)

    def drelu(self, a):
        ## Gives gradient wrt a of ReLu function
        return 1.0*(a>0)

    def sigmoid(self, b):
        return 1 / (1+ np.exp(- b))



    def fit(self, X, y):
        m, n = X.shape
        V = np.random.randn(n, self.units)
        W = np.random.randn(self.units) 
        
        for i in range(self.ite):
            a = X.dot(V)
            z = self.relu(a)
            b = z.dot(W)
            y_pred = self.sigmoid(b) 

            delta2 = y_pred - y
            theta2 = z.T.dot(delta2) + self.reg * W
            delta1 = np.outer(delta2,W.T) * (self.drelu(z)) ## Element-wise multiplication here, NOT matrix multiplication!
            theta1 = X.T.dot(delta1) + self.reg * V 
            W = W - self.ss * theta2
            V = V - self.ss * theta1
        return V, W


    def predprob(self,V,W, X):
        
        a = X.dot(V)
        z = self.relu(a)
        b = z.dot(W)
        y_pred_prob = self.sigmoid(b)
        return y_pred_prob


    def pred(self, V,W, X):
        
        a = X.dot(V)
        z = self.relu(a)
        b = z.dot(W)
        y_pred = self.sigmoid(b)>0.5
        return y_pred


    def cross_entropy(self, V,W, X, y):
        return log_loss(y.astype('str'), self.predprob(V,W,X))

    def errors(self, V,W, X, y):
        return 1- accuracy_score( y, self.pred(V,W,X))
'''
    def plot(self, X, y, Xt, yt, metrics = 'cross_entropy', parameter = 'step_size', parameter_value):
        if parameter == 'step_size':
            for p in parameter_value:
                m = cs.Neural_Network(10, p, 1000, 0.1)
                y_pred = m.pred(X)

'''



if __name__ == '__main__':
    pass
else:
    print("The package has been successfully loaded!")
 
