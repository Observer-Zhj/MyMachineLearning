# -*- coding:utf-8 _*-

"""
@author: yufu
@file: svm
@time: 2020/05/28
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def single_rbf_kernel(x, y, sigma):
    tmp = x - y
    if len(y.shape) == 1:
        return np.exp(-np.dot(tmp, tmp) / (2 * sigma ** 2))
    return np.exp(-np.sum(tmp ** 2, axis=1) / (2 * sigma ** 2))


def rbf_kernel(X, sigma):
    n = len(X)
    res = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            tmp = X[i] - X[j]
            tmp = np.dot(tmp, tmp)
            res[i, j] = tmp
            res[j, i] = tmp
    return np.exp(-res / (2 * sigma ** 2))


def single_linear_kernel(x, y):
    return np.matmul(y, x)


def linear_kernel(X):
    return np.matmul(X, X.T)


class SVM:
    def __init__(self, C=1, max_iter=100, kernel="rbf", sigma=10, pre_compute=True, tol=0.0001):
        self.C = C
        self.max_iter = max_iter
        self.kernel = kernel
        self.sigma = sigma
        self.pre_conpute = pre_compute
        self.tol = tol
        self.alpha = None
        self.b = 0
        self.support_vector = None
        self.support_alpha = None
        self.support_y = None
    
    def cal_kernel(self, X):
        if self.kernel == "rbf":
            return rbf_kernel(X, self.sigma)
        if self.kernel == "linear":
            return linear_kernel(X)
        raise ValueError("kernel must be 'rbf' or 'linear'")
    
    def cal_single_kernel(self, X):
        if self.kernel == "rbf":
            return single_rbf_kernel(X, self.support_vector, self.sigma)
        if self.kernel == "linear":
            return single_linear_kernel(X, self.support_vector)
        raise ValueError("kernel must be 'rbf' or 'linear'")
    
    def cal_g(self, y, alpha, kernel):
        return np.sum(alpha * y * kernel) + self.b
    
    def is_satisfy_kkt(self, alpha_1, yg):
        if (np.abs(alpha_1) < 1e-4) and (yg >= 1):
            return True
        elif (np.abs(alpha_1 - self.C) < 1e-4) and (yg <= 1):
            return True
        elif (alpha_1 > -1e-4) and (alpha_1 < (self.C + 1e-4)) \
                and (np.abs(yg - 1) < 1e-4):
            return True

        return False
    
    def cal_e(self, g, y):
        return g - y
    
    def get_a2(self, a1, E):
        e1 = E[a1]
        a2 = -1
        if e1 >= 0:
            prev = np.inf
            for i in range(len(E)):
                if i == a1:
                    continue
                if E[i] < prev:
                    a2 = i
                    prev = E[i]
        else:
            prev = -np.inf
            for i in range(len(E)):
                if i == a1:
                    continue
                if E[i] > prev:
                    a2 = i
                    prev = E[i]
        return a2
    
    def _update(self, a1, E, G, kernel_mat):
        e1 = E[a1]
        a2 = self.get_a2(a1, E)
        alpha_1_old = self.alpha[a1]
        alpha_2_old = self.alpha[a2]
        e2 = E[a2]
        if y[a1] != y[a2]:
            L = max(0, alpha_2_old - alpha_1_old)
            H = min(0, alpha_2_old - alpha_1_old) + self.C
        else:
            L = max(0, alpha_2_old + alpha_1_old - self.C)
            H = min(self.C, alpha_2_old + alpha_1_old)
    
        eta = kernel_mat[a1, a1] + kernel_mat[a2, a2] - 2 * kernel_mat[a1, a2]
        alpha_2_new = np.clip(alpha_2_old + y[a2] * (e1 - e2) / eta, L, H)
        alpha_1_new = alpha_1_old + y[a1] * y[a2] * (self.alpha[a2] - alpha_2_new)
        self.alpha[a1] = alpha_1_new
        self.alpha[a2] = alpha_2_new
    
        if np.abs(self.alpha[a2] - alpha_2_old) < self.tol:
            return 1
    
        b1 = -e1 - y[a1] * kernel_mat[a1, a1] * (alpha_1_new - alpha_1_old) - y[a2] * kernel_mat[
            a2, a1] * (alpha_2_new - alpha_2_old) + self.b
    
        b2 = -e2 - y[a1] * kernel_mat[a1, a2] * (alpha_1_new - alpha_1_old) - y[a2] * kernel_mat[
            a2, a2] * (alpha_2_new - alpha_2_old) + self.b
    
        if 0 < self.alpha[a1] < self.C:
            self.b = b1
        elif 0 < self.alpha[a2] < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2
        support_index = np.where(self.alpha > 0)[0]
        G[a1] = self.cal_g(y[support_index], self.alpha[support_index], kernel_mat[a1, support_index])
        G[a2] = self.cal_g(y[support_index], self.alpha[support_index], kernel_mat[a2, support_index])
        E[a1] = G[a1] - y[a1]
        E[a2] = G[a2] - y[a2]
        return 0
    
    def fit(self, X, y):
        self.alpha = np.zeros(len(X))
        kernel_mat = self.cal_kernel(X)
        G = np.sum(self.alpha * y * kernel_mat, axis=1) + self.b
        E = G - y
        for i in range(self.max_iter):
            alpha_change = False
            for a1 in range(len(self.alpha)):
                alpha_1_old = self.alpha[a1]
                g = G[a1]
                yg = y[a1] * g
                if not self.is_satisfy_kkt(alpha_1_old, yg):
                    alpha_change = True
                else:
                    continue
                status = self._update(a1, E, G, kernel_mat)
                if status == 0:
                    break
            if not alpha_change:
                break
        support_index = np.where(self.alpha > 0)[0]
        self.support_vector = X[support_index]
        self.support_alpha = self.alpha[support_index]
        self.support_y = y[support_index]
        return self
        
    def _predict(self, X):
        kernel = self.cal_single_kernel(X)
        return np.sum(self.support_alpha * self.support_y * kernel) + self.b
    
    def decision_function(self, X):
        return np.array([self._predict(i) for i in X])
    
    def predict(self, X):
        res = [1 if i >= 0 else -1 for i in self.decision_function(X)]
        return np.array(res)


bc = load_breast_cancer()
y = (bc.target - 0.5) * 2
maxx = np.max(bc.data, axis=0)
minx = np.min(bc.data, axis=0)
data = (bc.data - minx) / (maxx - minx)

train_X, test_X, train_y, test_y = train_test_split(data, y, test_size=0.3, random_state=5)

svm = SVM(kernel="rbf", sigma=10, max_iter=100, tol=0.01)
svm.fit(train_X, train_y)
y_pred = svm.predict(test_X)
print(confusion_matrix(test_y, y_pred))

print(svm.support_y * svm.decision_function(svm.support_vector))


svm = SVM(kernel="linear", max_iter=100)
svm.fit(train_X, train_y)
y_pred = svm.predict(test_X)
print(confusion_matrix(test_y, y_pred))

print(svm.support_y * svm.decision_function(svm.support_vector))
