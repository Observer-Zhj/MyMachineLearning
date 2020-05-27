# -*- coding: utf-8 -*-
# @Author   : ZhengHj
# @Time     : 2018/12/13 20:31
# @File     : linear_regression.py
# @Software : PyCharm


import os
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr
import matplotlib.pyplot as plt
from linear_model.auxiliary import MyShuffle
from linear_model.auxiliary import normalize_feature


class LinearRegression:
    
    def __init__(self, fit_intercept=True, normalize=False, method='normal', maxit=1000, alpha=0.1, 
                 batch=20, tol=1e-8):
        self._fit_intercept = fit_intercept
        self._normalize = normalize
        self._method = method.upper()
        if self._method == 'GD':
            self._maxit = maxit
            self._alpha = alpha
            self._batch = batch
            self._tol = tol
    
    def fit(self, X, y):
        X = np.mat(X, dtype='float64')
        y = np.mat(y, dtype='float64').T
        if self._fit_intercept:
            intercept = np.ones((X.shape[0], 1))
            X = np.concatenate((intercept, X), axis=1)
        if self._normalize:
            X_norm = np.mat(np.linalg.norm(X, ord=2, axis=0))
            X = X/X_norm
            y_norm = np.linalg.norm(y, ord=2, axis=0)
            y = y/y_norm
        if self._method == 'NORMAL':
            if sparse.issparse(X):
                # 如果`X`是稀疏矩阵，用QR算法求解最小二乘法。
                theta = self.qr(X, y)
            else:
                # 用SVD求解。
                theta = self.svd_method(X, y)
            self._cost = np.array(self.square_cost(X, y, theta)).squeeze()
            if self._normalize:
                theta = theta/X_norm.T*y_norm
            self._theta = theta
            
        elif self._method == 'GD':
            # 用批梯度下降求解
            self._theta, self._cost = self.gradient_decent(X, y, self._maxit, self._alpha, self._batch, self._tol)
            if self._normalize:
                self._theta = self._theta/X_norm.T*y_norm
        else:
            raise ValueError('method should be one of [normal, gd]')
        return self._theta
    
    def qr_method(self, X, y):
        theta = np.mat(lsqr(X, y)[0])
        theta.shape = (-1,1)
        return theta
    
    def svd_method(self, X, y):
        u, s, v = np.linalg.svd(X.T*X)
        l1 = np.where(s == 0)[0]
        s[l1] = 0.0001
        l = np.where(s < 0.001)[0]
        s = 1/s
        s[l] = 0
        s = np.mat(np.diag(s))
        theta = v.T*s*u.T*X.T*y
        return theta
    
    @classmethod
    def square_cost(cls, X, y, theta):
        residual  = X*theta - y
        cost = residual.T*residual / (2*X.shape[0])
        return cost.tolist()[0][0]
    
    @classmethod
    def gradient(cls, X, y, theta):
        g = X.T*(X*theta - y)/X.shape[0]
        return g
    
    @classmethod
    def gradient_decent(cls, X, y, maxit=1000, alpha=0.1, batch=20, tol=1e-8):
        theta = np.mat(np.random.normal(0, 1, X.shape[1])).T
        cost = [cls.square_cost(X, y, theta)]
        dt = MyShuffle([X, y])
        
        for i in range(maxit):
            inp, out = dt.next_batch(batch)
            g = cls.gradient(inp, out, theta)
            theta = theta - alpha*g
            cost.append(cls.square_cost(X, y, theta))
            if np.abs(cost[-1]-cost[-2]) < tol:
                break
            
        return theta, cost
    
    def predict(self, X):
        X = np.mat(X, dtype='float64')
        if self._fit_intercept:
            self._predictions = X*self._theta[1:] + self._theta[0]
        else:
            self._predictions = X*self._theta
        return self._predictions
    
    
#    def significance_test(self):
#        n, k = self._X.shape
#        X = self._X
#        y = np.mat(self._y).T
#        y_m = y.mean()
#        y_ = self.predict(X)
#        SST = np.multiply(y-y_m, y-y_m).sum()
#        SSR = np.multiply(y_-y_m, y_-y_m).sum()
#        SSe = np.multiply(y-y_, y-y_).sum()
#        
#        F = (SSR/k)/(SSe/(n-k-1))
#        r2 = F/(F+(n-k-1))
#        
#        sigma_ = sqrt(SSe/(n-k-1))
#        X_m = X.mean(0)
#        lxx = [(a-b)*(a-b).T for a,b in zip(X.T, X_m.T)]


def main():
    df = pd.read_csv("data2.txt", names=['square', 'bedrooms', 'price'])
    # df = normalize_feature(df)
    X = np.mat(df.iloc[:, :-1])
    y = np.array(df.iloc[:, -1])
    reg = LinearRegression(fit_intercept=False, normalize=True, method='normal')
    # reg = LinearRegression(fit_intercept=False, normalize=True, method='gd', maxit=10000, alpha=0.1, batch=20, tol=1e-8)
    reg.fit(X, y)
    pre = reg.predict(X)
    p = np.array(pre).squeeze()
    err = p - y
    cost = 0.5*(err*err).mean()
    plt.figure()
    plt.plot(reg._cost)
    print(cost)
    print(reg._theta)
    plt.figure()
    plt.plot(p, color='red')
    plt.plot(y, color='black')
    plt.show()


if __name__ == '__main__':
    main()

