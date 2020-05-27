# -*- coding:utf-8 _*-

import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.rdd import RDD
from datetime import datetime
from sklearn.datasets import load_iris

conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf=conf)


def cal_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


class KMeansOnSpark:
    def __init__(self, n_clusters=3, max_iter=100, seed=None):
        self.n_cluseters = n_clusters
        self.max_iter = max_iter
        self.seed = seed
        self.clusters = None
    
    def _init_clusters(self, data):
        return data.takeSample(False, self.n_cluseters, self.seed)
    
    def fit(self, data: RDD):
        # data.cache()
        # 初始化聚类中心
        clusters = self._init_clusters(data)
        for i in range(self.max_iter):
            # 计算data和聚类中心的距离
            dists = data.map(lambda x: (x, [cal_distance(x, i) for i in clusters]))
            # 找到每条数据最近的聚类中心
            tmp = dists.map(lambda x: (np.argmin(x[1]), x[0]))
            # 生成新的聚类中心
            clusters = tmp.mapValues(lambda x: (x, 1)).reduceByKey(
                lambda x, y: (x[0] + y[0], x[1] + y[1])).map(
                lambda x: x[1][0] / x[1][1]).collect()
        self.clusters = clusters
        return self
    
    def predict(self, data: RDD):
        dists = data.map(lambda x: (x, [cal_distance(x, i) for i in self.clusters]))
        tmp = dists.map(lambda x: np.argmin(x[1]))
        return tmp.collect()


iris = load_iris()
data = sc.parallelize(iris.data)

km = KMeansOnSpark(3, 100)
t1 = datetime.now()
km.fit(data)
t2 = datetime.now()
print(t2 - t1)
print(km.predict(data))