
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import accuracy_score

class Logistic_Regression(object):	

	def __init__(self, step = 0.005, l2 = 1, iteration = 1000 ):
		self.step = step
		self.l2 = l2
		self.iteration = iteration

	def u(self, theta, X):
		return 1 / (1+ np.exp(-(np.dot(X, theta))))

	def g(self, theta, X, y):
		m = X.shape[0]
		return 1.0/m * np.dot(X.T, self.u(theta, X)-y)

	def plot(self, X, y):
		n = X.shape[1]
		ite = self.iteration
		theta0 = np.zeros(n)
		theta = theta0.reshape(1,-1)
		for i in range(ite):
			try:
				theta1 = theta0 - self.step * (self.g(theta0, X,y) + self.l2 * theta0)
				theta = np.concatenate((theta, theta1.reshape(1,-1)))			
				theta0 = theta1
			except:
				print('Error on %dth try' % i)
		pd.DataFrame(theta).plot()
		plt.show()

	def pred(self,X_train, y_train, X_test, threshold):
		n = X_train.shape[1]
		ite = self.iteration
		theta0 = np.zeros(n)
		for i in range(ite):
			theta1 = theta0 - self.step * (self.g(theta0, X_train, y_train) + self.l2 * theta0)
			theta0 = theta1
		y_pred_proba = 1 / (1 + np.exp(-np.dot(X_test, theta0)))
		y_pred = (y_pred_proba > threshold).astype('int64')
		return y_pred
	def theta(self, X_train, y_train):
		n = X_train.shape[1]
		ite = self.iteration
		theta0 = np.zeros(n)
		for i in range(ite):
			theta1 = theta0 - self.step * (self.g(theta0, X_train, y_train) + self.l2 * theta0)
			theta0 = theta1
		return theta0
	def predprob(self,X_train, y_train, X_test):
		n = X_train.shape[1]
		ite = self.iteration
		theta0 = np.zeros(n)
		for i in range(ite):
			theta1 = theta0 - self.step * (self.g(theta0, X_train, y_train) + self.l2 * theta0)
			theta0 = theta1
		y_pred_proba = 1 / (1 + np.exp(-np.dot(X_test, theta0)))
		return y_pred_proba
	def score(self,X_train, y_train, X_test, y_test,threshold):
		return accuracy_score(self.pred(X_train, y_train, X_test, threshold), y_test)

if __name__ == '__main__':
    pass
 
