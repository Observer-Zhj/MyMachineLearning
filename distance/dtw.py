# -*- coding: utf-8 -*-
# @Author   : ZhengHj
# @Time     : 2019/5/15 22:54
# @Project  : testProject
# @File     : distance.py
# @Software : PyCharm

import numpy as np


def dtw_distance(ts_a, ts_b, d=lambda x, y: abs(x - y)):
    """计算两个序列的DTM距离

    Args:
        ts_a: iterable
        ts_b: iterable
        d: distance function

    Returns:
        dtw distance
    """

    # 创建距离矩阵t
    ts_a, ts_b = np.array(ts_a), np.array(ts_b)
    M, N = len(ts_a), len(ts_b)
    cost = np.ones((M, N))

    # 初始化第一行和第一列
    cost[0, 0] = d(ts_a[0], ts_b[0])
    for i in range(1, M):
        cost[i, 0] = cost[i - 1, 0] + d(ts_a[i], ts_b[0])

    for j in range(1, N):
        cost[0, j] = cost[0, j - 1] + d(ts_a[0], ts_b[j])

    # 计算矩阵剩下的值
    for i in range(1, M):
        for j in range(1, N):
            choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])
    # 返回DTW值
    return cost[-1, -1]