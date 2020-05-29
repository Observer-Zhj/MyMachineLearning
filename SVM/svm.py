# -*- coding:utf-8 _*-

"""
@author: yufu
@file: svm
@time: 2020/05/28
"""

import numpy as np
from datetime import datetime
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
    """
    pre_compute设置是否预计算核矩阵
    """
    def __init__(self, C=1, max_iter=100, kernel="rbf", sigma=10, pre_compute=True, tol=0.0001):
        self.C = C
        self.max_iter = max_iter
        self.kernel = kernel
        self.sigma = sigma
        self.pre_compute = pre_compute
        self.tol = tol
        self.alpha = None
        self.b = 0
        self.support_vector = None
        self.support_alpha = None
        self.support_y = None
    
    def cal_kernel(self, X):
        """ 计算核矩阵 """
        if self.kernel == "rbf":
            return rbf_kernel(X, self.sigma)
        if self.kernel == "linear":
            return linear_kernel(X)
        raise ValueError("kernel must be 'rbf' or 'linear'")
    
    def cal_single_kernel(self, X, Y):
        """ 计算一个向量X和一个向量/矩阵Y的核 """
        if self.kernel == "rbf":
            return single_rbf_kernel(X, Y, self.sigma)
        if self.kernel == "linear":
            return single_linear_kernel(X, Y)
        raise ValueError("kernel must be 'rbf' or 'linear'")
    
    def cal_g(self, y, alpha, kernel):
        return np.sum(alpha * y * kernel) + self.b
    
    def cal_G(self, X, y, kernel):
        if kernel is None:
            return np.array(
                [np.sum(self.alpha * y * self.cal_single_kernel(X[i], X)) for i in range(len(self.alpha))]) + self.b
        else:
            return np.sum(self.alpha * y * kernel, axis=1) + self.b
    
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
    
    def get_K(self, kernel_mat, a1, a2, X):
        if kernel_mat is None:
            tmp = self.cal_kernel(X[[a1, a2]])
            return tmp[0, 0], tmp[0, 1], tmp[1, 0], tmp[1, 1]
        return kernel_mat[a1, a1], kernel_mat[a1, a2], kernel_mat[a2, a1], kernel_mat[a2, a2]
    
    def _update(self, X, a1, E, G, kernel_mat):
        e1 = E[a1]
        a2 = self.get_a2(a1, E)
        alpha_1_old = self.alpha[a1]
        alpha_2_old = self.alpha[a2]
        e2 = E[a2]
        K11, K12, K21, K22 = self.get_K(kernel_mat, a1, a2, X)
        if y[a1] != y[a2]:
            L = max(0, alpha_2_old - alpha_1_old)
            H = min(0, alpha_2_old - alpha_1_old) + self.C
        else:
            L = max(0, alpha_2_old + alpha_1_old - self.C)
            H = min(self.C, alpha_2_old + alpha_1_old)
    
        eta = K11 + K22 - 2 * K12
        alpha_2_new = np.clip(alpha_2_old + y[a2] * (e1 - e2) / eta, L, H)
        alpha_1_new = alpha_1_old + y[a1] * y[a2] * (self.alpha[a2] - alpha_2_new)
        self.alpha[a1] = alpha_1_new
        self.alpha[a2] = alpha_2_new
    
        if np.abs(self.alpha[a2] - alpha_2_old) < self.tol:
            return 1
        print(a1, a2)
        b1 = -e1 - y[a1] * K11 * (alpha_1_new - alpha_1_old) - y[a2] * K21 * (alpha_2_new - alpha_2_old) + self.b
    
        b2 = -e2 - y[a1] * K12 * (alpha_1_new - alpha_1_old) - y[a2] * K22 * (alpha_2_new - alpha_2_old) + self.b
    
        if 0 < self.alpha[a1] < self.C:
            self.b = b1
        elif 0 < self.alpha[a2] < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2
        support_index = np.where(self.alpha > 0)[0]
        G[a1] = self.cal_g(y[support_index], self.alpha[support_index], self.cal_single_kernel(X[a1], X[support_index]))
        G[a2] = self.cal_g(y[support_index], self.alpha[support_index], self.cal_single_kernel(X[a2], X[support_index]))
        E[a1] = G[a1] - y[a1]
        E[a2] = G[a2] - y[a2]
        return 0
    
    def fit(self, X, y):
        self.alpha = np.zeros(len(X))
        if self.pre_compute:
            kernel_mat = self.cal_kernel(X)
        else:
            kernel_mat = None
        G = self.cal_G(X, y, kernel_mat)
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
                status = self._update(X, a1, E, G, kernel_mat)
                if status == 0:
                    break
                alpha_change = False
            if not alpha_change:
                break
        support_index = np.where(self.alpha > 0)[0]
        self.support_vector = X[support_index]
        self.support_alpha = self.alpha[support_index]
        self.support_y = y[support_index]
        return self
        
    def _predict(self, X):
        kernel = self.cal_single_kernel(X, self.support_vector)
        return self.cal_g(self.support_y, self.support_alpha, kernel)
    
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

train_X, test_X, train_y, test_y = train_test_split(data, y, test_size=0.3)


svm = SVM(kernel="rbf", sigma=10, max_iter=100, pre_compute=True)
t1 = datetime.now()
svm.fit(train_X, train_y)
t2 = datetime.now()
print(t2 - t1)
y_pred = svm.predict(test_X)
print(confusion_matrix(test_y, y_pred))

print(svm.support_y * svm.decision_function(svm.support_vector))


svm = SVM(kernel="linear", max_iter=100, pre_compute=True)
t1 = datetime.now()
svm.fit(train_X, train_y)
t2 = datetime.now()
print(t2 - t1)
y_pred = svm.predict(test_X)
print(confusion_matrix(test_y, y_pred))

print(svm.support_y * svm.decision_function(svm.support_vector))
