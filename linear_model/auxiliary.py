# -*- coding: utf-8 -*-
# @Author   : ZhengHj
# @Time     : 2018/12/13 27:54
# @File     : auxiliary.py
# @Software : PyCharm


import numpy as np
import pandas as pd


class MyShuffle:
    """随机不重复抽样。
    
    Args:
        x: iterable object(list,tuple..)。x的格式为`[x1,x2,...,xn]`（以list为例，一般都是对
        `[input_data, output_data]`操作），其中列表中的每个元素xi必须是`ndarray`，且它们的的第一个维度都相同。
        每次打乱数据都是打乱`x1`的第一维，后面所有的`xi`也按照`x1`打乱的顺序打乱。
    """
    def __init__(self, x):
    
        self._x = x
        self._is_shuffling = 0
        self._end = 0
        self._l = len(x[0])
        
    def next_batch(self, batch):
        """
        打乱数据并生成一个迭代器。
    
        数据打乱的规则见初始化函数的参数介绍。
        如果是第一次抽样，则会先打乱数据，然后从中抽取一个`batch`的数据。后面每次调用`next_batch`方法都会在打乱后的数据中
        取出一个`batch`大小的数据。在一轮数据被取完前，取出的数据不会重复。如果取到最后剩余数据小于`batch`，则全部取出，再
        一次抽样的时候所有数据重新打乱。
    
        Args:
            batch：int, 每次抽样取出的数据样本个数。
            
        Returns:
            抽样后的数据。
        """
        if self._is_shuffling == 0:
            self._randomSeries = np.random.permutation(self._l)
            self._x_shuffle = [i[self._randomSeries, ] for i in self._x]
            self._is_shuffling = 1
        
        end = min(self._end+batch, self._l)
        r = np.arange(self._end, end)
        out = [i[r, ] for i in self._x_shuffle]
        self._end += batch
        if self._end >= self._l:
            self._end = 0
            self._is_shuffling = 0
            
        return out


def normalize_feature(df):
    return df.apply(lambda column: (column - column.mean()) / column.std())


def feature_mapping(x, y, power, as_ndarray=False):
    data = {"f{}{}".format(i - p, p): np.power(x, i - p) * np.power(y, p)
                for i in np.arange(power + 1)
                for p in np.arange(i + 1)
            }

    if as_ndarray:
        return pd.DataFrame(data).as_matrix()
    else:
        return pd.DataFrame(data)