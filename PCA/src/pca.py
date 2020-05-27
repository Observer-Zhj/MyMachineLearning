# -*- coding: utf-8 -*-
# @Author   : ZhengHj
# @Time     : 2019/5/11 19:07
# @File     : pca.py
# @Software : PyCharm


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 生成一个二维正态分布，协方差矩阵为[[1, 0.5], [0.5, 1]]，并在x轴上拉伸10倍

np.random.seed(0)
n = 10000
mu = [0, 0]
cov = np.array([[1, 0.5], [0.5, 1]])
X = np.random.multivariate_normal(mu, cov, n)
xd1 = np.linspace(-40, 40, 100)
xd2 = np.linspace(-4, 4, 10)

X = X * [10, 1]
scaler = StandardScaler()
scaler.fit(X)
X_standard = scaler.transform(X)

# 数据展示
plt.figure()
plt.subplot(211)
plt.plot(xd1, xd1 / 10, "--k")
plt.scatter(X[:, 0], X[:, 1])
plt.title("raw data")
plt.subplot(212)
plt.plot(xd2, xd2, "--k")
plt.scatter(X_standard[:, 0], X_standard[:, 1])
plt.title("standard data")
plt.show()

# 创建PCA对象并训练
# 使用原始数据
model1 = PCA()
model1.fit(X)
# 使用z-score之后的数据
model2= PCA()
model2.fit(X_standard)

print("principal component: \n{}".format(model2.components_))
print("principal component variance: \n{}".format(model2.explained_variance_))
print("principal component variance ratio in total variance: \n{}".format(model2.explained_variance_ratio_))
print("principal component: \n{}".format(model2.transform(X_standard)))

# 比较数据经过z-score标准化前后的PCA结果，很明显，未经过标准化的数据的第一主成分会往量纲大的方向偏移
plt.figure()
# 原始数据的PCA结果
plt.subplot(211)
plt.scatter(X[:, 0], X[:, 1])
components1 = model1.components_
variance1 = np.sqrt(model1.explained_variance_)
plt.plot(xd1, xd1 / 10, "--k")
plt.arrow(0, 0, components1[0, 0]*variance1[0], components1[0, 1]*variance1[0], color='red', head_width=.3)
plt.arrow(0, 0, components1[1, 0]*variance1[1], components1[1, 1]*variance1[1], color='blue', head_width=.3)
plt.title("raw data PCA")
# 经过标准化的数据的PCA结果
plt.subplot(212)
plt.scatter(X_standard[:, 0], X_standard[:, 1])
components2 = model2.components_
variance2 = np.sqrt(model2.explained_variance_)
plt.plot(xd2, xd2, "--k")
plt.arrow(0, 0, components2[0, 0]*variance2[0], components2[0, 1]*variance2[0], color='red', head_width=.3)
plt.arrow(0, 0, components2[1, 0]*variance2[1], components2[1, 1]*variance2[1], color='blue', head_width=.3)
plt.title("standard data PCA")
plt.show()
