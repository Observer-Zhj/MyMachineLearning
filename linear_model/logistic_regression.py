# -*- coding: utf-8 -*-
# @Author   : ZhengHj
# @Time     : 2018/12/13 21:39
# @File     : logistic_regression.py
# @Software : PyCharm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linear_model.auxiliary import MyShuffle, feature_mapping
from sklearn.metrics import classification_report


class LogisticRegression:
    
    def __init__(self, fit_intercept=True, maxit=1000, alpha=0.1, batch=20, tol=1e-8, regularization=False, l=1):
        self._fit_intercept = fit_intercept
        self._maxit = maxit
        self._alpha = alpha
        self._batch = batch
        self._tol = tol
        self._regularization = regularization
        self._l = l
    
    def fit(self, X, y):
        X = np.mat(X, dtype='float64')
        y = np.mat(y, dtype='float64').T
        if self._fit_intercept:
            intercept = np.ones((X.shape[0], 1))
            X = np.concatenate((intercept, X), axis=1)
        self._theta, self._cost = self.gradient_decent(X, y, self._maxit, self._alpha, self._batch, self._tol)
        return self._theta
    
    @classmethod
    def sigmoid(cls, X, theta):
        z = X*theta
        return 1 / (1+np.exp(-z))
    
    def cross_entropy_cost(self, X, y, theta):
        cost = (-y.T*np.log(self.sigmoid(X, theta)) - (1-y).T*np.log(1-self.sigmoid(X,theta)))/X.shape[0]
        return cost.mean()
    
    def regularized_cost(self, X, y, theta, l=1, fit_intercept=True):
        if fit_intercept:
            theta1 = theta[1:]
        cost = l / (2*X.shape[1]) * np.power(theta1, 2).sum()
        return self.cross_entropy_cost(X, y, theta) + cost
    
    def gradient(self, X, y, theta):
        g = X.T*(self.sigmoid(X, theta) - y)/X.shape[0]
        return g
    
    def regularized_gradient(self, X, y, theta):
        if self._fit_intercept:
            theta1 = theta[1:]
            gradient = self._l / X.shape[1] * theta1
            gradient = np.concatenate(([[0]], gradient))
        else:
            gradient = self._l / X.shape[1] * theta
        return self.gradient(X, y, theta) + gradient
    
    def gradient_decent(self, X, y, maxit=1000, alpha=0.1, batch=20, tol=1e-8):
        theta = np.mat(np.random.normal(0, 1, X.shape[1])).T
        # theta = np.mat(np.zeros((X.shape[1], 1)))
        cost = [self.cross_entropy_cost(X, y, theta)]
        dt = MyShuffle([X, y])
        if self._regularization:
            gradient_func = self.regularized_gradient
        else:
            gradient_func = self.gradient
        
        for i in range(maxit):
            inp, out = dt.next_batch(batch)
            g = gradient_func(X, y, theta)
            theta = theta - alpha*g
            cost.append(self.cross_entropy_cost(X, y, theta))
            if np.abs(cost[-1]-cost[-2]) < tol:
                break
            
        return theta, cost
    
    def predict(self, X):
        X = np.mat(X, dtype='float64')
        if self._fit_intercept:
            intercept = np.ones((X.shape[0], 1))
            X = np.concatenate((intercept, X), axis=1)
        pre = self.sigmoid(X, self._theta)
        self._predictions = [1 if x >= 0.5 else 0 for x in pre]
        
        return self._predictions


def main():
    df = pd.read_csv("data4.txt", names=['test1', 'test2', 'accepted'])
    
    positive = df[df['accepted'].isin([1])]
    negative = df[df['accepted'].isin([0])]
    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(positive['test1'], positive['test2'], s=50, c='b', marker='o', label='Accepted')
    ax.scatter(negative['test1'], negative['test2'], s=50, c='r', marker='x', label='Rejected')
    ax.legend()
    ax.set_xlabel('test1 Score')
    ax.set_ylabel('test2 Score')
    plt.show()
    
    x1 = np.array(df.test1)
    x2 = np.array(df.test2)
    X = feature_mapping(x1, x2, 6, False)
    y = np.array(df.accepted)
    reg = LogisticRegression(fit_intercept=True, maxit=10000, alpha=0.1, tol=1e-8, regularization=True, l=0.5)
    reg.fit(X, y)
    print(reg._theta)
    pre = reg.predict(X)
    print(classification_report(y, pre))
    plt.figure()
    plt.plot(reg._cost)
    plt.show()


if __name__ == '__main__':
    main()
