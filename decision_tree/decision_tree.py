# -*- coding:utf-8 _*-

"""
@author: yufu
@file: decision_tree
@time: 2020/04/30
"""

import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
from abc import ABC, abstractmethod
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class BaseNode(ABC):
    def __init__(self, depth):
        self.feature_index = None
        self.split = None
        self.value = None
        self.stop = False
        self.left = None
        self.right = None
        self.depth = depth
        self.is_leaf = False
        self.left_value = None
        self.right_value = None
    
    @abstractmethod
    def _get_best_split(self, feature, label):
        raise NotImplementedError
    
    @abstractmethod
    def get_best_split(self, data, label):
        raise NotImplementedError
    
    def predict(self, data):
        if self.is_leaf:
            return self.value
        if data[self.feature_index] < self.split:
            if self.left:
                return self.left.predict(data)
            return self.left_value
        elif self.right:
            return self.right.predict(data)
        return self.right_value


class RegressionNode(BaseNode):
    def __init__(self, depth):
        super().__init__(depth=depth)
    
    def _get_best_split(self, feature, label):
        length = len(feature)
        if len(np.unique(feature)) == 1:
            c = np.mean(label)
            return feature[0], np.sum((label - c) ** 2), c, c, c
        sort_indices = np.argsort(feature)
        feature = feature[sort_indices]
        label = label[sort_indices]
        best_split = None
        best_c1 = None
        best_c2 = None
        min_loss = -np.sum(label) ** 2 / length
        prev = feature[0]
        sm = np.sum(label)
        left_sm = label[0]
        left_square = label[0] ** 2
        for i in range(1, length):
            if feature[i] != prev:
                c_1 = left_sm / i
                c_2 = (sm - left_sm) / (length - i)
                loss_1 = -c_1 ** 2 * i
                loss_2 = -c_2 ** 2 * (length - i)
                loss = loss_1 + loss_2
                if loss < min_loss:
                    min_loss = loss
                    best_split = (prev + feature[i]) / 2
                    best_c1 = c_1
                    best_c2 = c_2
                prev = feature[i]
            left_sm += label[i]
            left_square += label[i] ** 2
        return best_split, min_loss, np.mean(label), best_c1, best_c2
    
    def get_best_split(self, data, label):
        loss = np.inf
        for i in range(data.shape[1]):
            split, _loss, _y, _c1, _c2 = self._get_best_split(data[:, i], label)
            if _loss < loss:
                self.feature_index = i
                self.split = split
                self.value = _y
                self.left_value = _c1
                self.right_value = _c2
                loss = _loss
            if self.split is None:
                self.is_leaf = True
            self.nums = len(label)


class ClassifierNode(BaseNode):
    def __init__(self, depth):
        super().__init__(depth=depth)
    
    def _get_best_split(self, feature, label):
        length = len(feature)
        left_value = defaultdict(int)
        right_value = Counter(label)
        value = max(right_value, key=right_value.get)
        if len(np.unique(feature)) == 1:
            gini = 1 - np.sum((np.array(list(right_value.values())) / length) ** 2)
            return feature[0], gini, value, value, value
        
        sort_indices = np.argsort(feature)
        feature = feature[sort_indices]
        label = label[sort_indices]
        best_split = None
        best_c1 = None
        best_c2 = None
        min_gini = np.inf
        prev = feature[0]
        prev_idx = 0
        for i in range(1, length):
            if feature[i] != prev:
                count = Counter(label[prev_idx:i])
                for k, v in count.items():
                    left_value[k] += v
                    right_value[k] -= v
                left_gini = 1 - np.sum((np.array(list(left_value.values())) / i) ** 2)
                right_gini = 1 - np.sum((np.array(list(right_value.values())) / (length - i)) ** 2)
                gini = i * left_gini / length + (length - i) * right_gini / length
                if gini < min_gini:
                    min_gini = gini
                    best_split = (prev + feature[i]) / 2
                    best_c1 = max(left_value, key=left_value.get)
                    best_c2 = max(right_value, key=right_value.get)
                prev = feature[i]
                prev_idx = i
        return best_split, min_gini, value, best_c1, best_c2
    
    def get_best_split(self, data, label):
        if len(np.unique(label)) == 1:
            self.value = label[0]
            self.is_leaf = True
        else:
            loss = np.inf
            for i in range(data.shape[1]):
                split, _loss, _y, _c1, _c2 = self._get_best_split(data[:, i], label)
                if _loss < loss:
                    self.feature_index = i
                    self.split = split
                    self.value = _y
                    self.left_value = _c1
                    self.right_value = _c2
                    loss = _loss


class RegressionTree:
    def __init__(self, max_depth=10, min_samples_leaf=1):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
    
    def _fit(self, root: RegressionNode, data, label):
        if len(data) == 0:
            return None
        if len(data) < self.min_samples_leaf:
            root.value = np.mean(label)
            root.is_leaf = True
            root.nums = len(label)
            return root
        root.get_best_split(data, label)
        if root.is_leaf:
            return root
        if root.depth >= self.max_depth:
            root.is_leaf = True
            return root
        index = data[:, root.feature_index] < root.split
        root.left = self._fit(RegressionNode(root.depth + 1), data[index], label[index])
        root.right = self._fit(RegressionNode(root.depth + 1), data[~index], label[~index])
        if root.left is None and root.right is None:
            root.is_leaf = True
        return root
    
    def fit(self, data, label):
        self.root = self._fit(RegressionNode(depth=1), data, label)
        return self
    
    def predict(self, data):
        res = []
        for i in range(len(data)):
            res.append(self.root.predict(data[i]))
        return np.array(res)


class ClassifierTree:
    def __init__(self, max_depth=10, min_samples_leaf=1):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
    
    def _fit(self, root: ClassifierNode, data, label):
        if root.depth >= self.max_depth or len(data) < self.min_samples_leaf:
            return None
        root.get_best_split(data, label)
        if root.is_leaf:
            return root
        index = data[:, root.feature_index] < root.split
        root.left = self._fit(ClassifierNode(root.depth + 1), data[index], label[index])
        root.right = self._fit(ClassifierNode(root.depth + 1), data[~index], label[~index])
        if root.left is None and root.right is None:
            root.is_leaf = True
        return root
    
    def fit(self, data, label):
        self.root = self._fit(ClassifierNode(depth=1), data, label)
        return self
    
    def predict(self, data):
        res = []
        for i in range(len(data)):
            res.append(self.root.predict(data[i]))
        return np.array(res)


class GBDTRegression:
    def __init__(self, learning_rate=0.1, n_estimators=100, max_depth=10, min_samples_leaf=1):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self._trees = []
        self._init = 0
    
    def fit(self, data, label):
        self._init = np.mean(label)
        y_ = self._init
        for i in range(self.n_estimators):
            tree = RegressionTree(self.max_depth, self.min_samples_leaf)
            tree.fit(data, label - y_)
            y_ += self.learning_rate * tree.predict(data)
            self._trees.append(tree)
        return self
    
    def predict(self, data):
        ans = [i.predict(data) for i in self._trees]
        return self._init + self.learning_rate * np.sum(ans, axis=0)


class GBDTClassifier:
    """ 二分类 """
    def __init__(self, learning_rate=0.1, n_estimators=100, max_depth=10, min_samples_leaf=1):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self._trees = []
        self._init = 0
        self.class_ = None
    
    def fit(self, data, label):
        enc = LabelEncoder()
        label = enc.fit_transform(label)
        self.class_ = enc.classes_
        self._init = np.mean(label)
        y_ = self._init
        for i in range(self.n_estimators):
            tree = RegressionTree(self.max_depth, self.min_samples_leaf)
            tree.fit(data, label - sigmoid(y_))
            y_ += self.learning_rate * tree.predict(data)
            self._trees.append(tree)
        return self
    
    def predict_proba(self, data):
        ans = [i.predict(data) for i in self._trees]
        return sigmoid(self._init + self.learning_rate * np.sum(ans, axis=0))
    
    def predict(self, data):
        ans = self.predict_proba(data)
        return self.class_[[0 if i < 0.5 else 1 for i in ans]]


boston = load_boston()
data = boston.data.copy()
label = boston.target.copy()

train_X, test_X, train_y, test_y = train_test_split(data, label, test_size=0.3, random_state=5)

dt = RegressionTree(max_depth=10, min_samples_leaf=4)
t1 = datetime.now()
dt.fit(train_X, train_y)
t2 = datetime.now()
print(t2 - t1)
y_pred = dt.predict(test_X)
print(np.mean((y_pred - test_y) ** 2))

gbdt = GBDTRegression(learning_rate=0.1, n_estimators=200, max_depth=5, min_samples_leaf=4)
t1 = datetime.now()
gbdt.fit(train_X, train_y)
t2 = datetime.now()
print(t2 - t1)
y_pred = gbdt.predict(test_X)
print(np.mean((y_pred - test_y) ** 2))



