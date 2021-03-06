# -*- coding:utf-8 _*-

"""
@author: yufu
@file: gmm
@time: 2020/05/26
"""

import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import norm


def pdf(x, mu, sigma):
    n = x.shape[-1]
    tmp1 = (2 * np.pi) ** (n / 2) * np.linalg.det(sigma) ** 0.5
    tmp1 = tmp1.reshape((-1, 1))
    x = x - mu[:, np.newaxis, :]
    tmp2 = np.sum(np.matmul(x, np.linalg.inv(sigma)) * x, axis=2)
    ans = 1 / tmp1 * np.exp(-0.5 * tmp2)
    return ans.T


class GMM:
    def __init__(self, n_clusters, max_iter=100, init="kmeans"):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init = init
        self.mean_ = None
        self.cov_ = None
        self.phi_ = None
    
    def init_clusters(self, X):
        self.cov_ = np.array([np.diag(np.ones(X.shape[1])) for i in range(self.n_clusters)])
        self.phi_ = np.ones(self.n_clusters) / self.n_clusters
        if self.init == "kmeans":
            self.mean_ = KMeans(self.n_clusters).fit(X).cluster_centers_
        elif self.init == "random":
            self.mean_ = np.random.random((self.n_clusters, X.shape[1]))
        else:
            raise ValueError("init param must be 'kmeans' or 'random'")
    
    def fit(self, X):
        self.init_clusters(X)
        for i in range(self.max_iter):
            weights = self._e_step(X)
            self._m_step(X, weights)
        return self
    
    def predict(self, X):
        return np.argmax(self._e_step(X), axis=1)
    
    def _e_step(self, X):
        weights = pdf(X, self.mean_, self.cov_) * self.phi_
        sums = np.sum(weights, axis=1)
        return weights / sums.reshape((-1, 1))
    
    def _m_step(self, X, weights):
        self.phi_ = np.mean(weights, axis=0)
        self.mean_ = self.update_mu(X, weights)
        self.cov_ = self.update_sigma(X, self.mean_, weights)
        
    def update_mu(self, X, w):
        shape = w.shape
        w = w.T.reshape((shape[1], shape[0], 1))
        return np.sum(w * X, axis=1) / np.sum(w, axis=1)
    
    def update_sigma(self, X, mu, w):
        shape = w.shape
        w = w.T.reshape((shape[1], shape[0], 1, 1))
        n = X.shape[-1]
        tmp = X[:, :, np.newaxis] - mu[:, np.newaxis, :, np.newaxis]
        tmp = np.matmul(tmp, tmp.reshape((shape[1], shape[0], 1, n)))
        return np.sum(tmp * w, axis=1) / np.sum(w, axis=1)


from sklearn.datasets import load_iris
from datetime import datetime

iris = load_iris()

model = GMM(3)
t1 = datetime.now()
model.fit(iris.data)
t2 = datetime.now()
print(t2 - t1)
print(model.predict(iris.data))
