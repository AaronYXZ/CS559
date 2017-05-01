import numpy as np
import scipy as sp
import pandas as pd



class HMM(object):
    def __init__(self, states = 3, observations = 10, length = 1000):
        self.N = states
        self.M = observations
        self.T = length

    def dataInitializer(self):
        ## http://stackoverflow.com/questions/18659858/generating-a-list-of-random-numbers-summing-to-1
        pi = np.random.dirichlet(np.ones(self.N) * 1000. , size = 1)
        A = np.empty((self.N,self.N))
        B = np.empty((self.N,self.M))
        for i in range(self.N):
            A[i] = np.random.dirichlet(np.ones(self.N) * 1000. , size = 1)
            B[i] = np.random.dirichlet(np.ones(self.M) * 1000. , size = 1)
        f = open('../../seq.txt')
        O = [int(x) for x in f.read().split()]
        return pi, A, B, O

    def forward(self, pi, A, B, O):
        alpha = np.zeros((self.T,self.N))
        c = np.zeros(self.T)

        alpha[0] = pi * B.T[0]
        c[0] = alpha[0].sum()

        for t in range(1, self.T):
            alpha[t] = (alpha[t-1] * A.T).sum(axis = 1)
            o = O[t]
            alpha[t] = alpha[t] * B.T[o-1]
            c[t] = alpha[t].sum()
            c[t] = 1/c[t]
            alpha[t] = alpha[t] * c[t]

        return c, alpha 

    def backward(self, c, A, B, O):
        beta = np.zeros((self.T, self.N))
        beta[self.T-1] = c[self.T-1]

        for x in range(1,self.T):
            t = self.T - 1 - x
            ob = O[t+1]
            beta[t] = (A*B.T[ob-1]*beta[t+1]).sum(axis = 1)
            beta[t] = beta[t] * c[t]
        return beta

    def gamma(self, A, B, O, alpha, beta):
        gamma = np.zeros((self.T, self.N))
        gamma2 = np.empty((self.T-1, self.N, self.N))
        for t in range(self.T-1):
            ob = O[t+1]
            denom = (alpha[t]*A*B.T[ob-1]*beta[t+1]).sum()
            gamma2[t] = (alpha[t] * A * B.T[ob-1]*beta[t+1]) / denom
            gamma[t] = gamma2[t].sum(axis = 1) 
        t = self.T-1
        denom2 = alpha[t].sum()
        gamma[t] = alpha[t] / denom2

        return gamma, gamma2

    def estimate(self, A, B, O, gamma, gamma2):
        pi = gamma[0]
        for i in range(self.N):
            for j in range(self.N):
                num = 0
                denom = 0
                for t in range(self.T-1):
                    num = num + gamma2[t][i][j]
                    denom = denom + gamma[t][i]
                A[i][j] = num / denom
        for i in range(self.N):
            for j in range(self.M):
                num = 0
                denom = 0
                for t in range(self.T):
                    ob = O[t]
                    if ob-1 == j:
                        num += gamma[t][i]
                    denom += gamma[t][i]
                B[i][j] = num / denom 
        return pi, A, B 
    def logProb(self, c):
        return - np.log(c).sum()
    def run(self, ite):
        oldLogProb = - 10**12
        i = 0
        pi, A, B, O = self.dataInitializer()
        while True:
            c, alpha = self.forward(pi, A, B, O)
            beta = self.backward(c, A, B, O)
            gamma, gamma2 = self.gamma(A, B, O, alpha, beta)
            pi, A, B = self.estimate(A, B, O, gamma, gamma2)
            logprob = self.logProb(c)
            if (logprob < oldLogProb) or (i > ite):
                print(i)
                return pi, A, B
            else:
                i+=1
                oldLogProb = logprob
        return pi, A, B 


if __name__ == '__main__':
    pass
else:
    print("The package has been successfully loaded!")