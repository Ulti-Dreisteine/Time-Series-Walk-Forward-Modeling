# -*- coding: utf-8 -*-
"""
Created on 2021/05/11 10:16:45

@File -> rf_walk_forward.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 使用Walk-forward建立预测模型 
"""

from sklearn.metrics import mean_squared_error as error
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(
    os.path.abspath(__file__), '../../../'))
sys.path.append(BASE_DIR)

from src.settings import *

# ---- Walk-forward模型验证 -------------------------------------------------------------------------


def walk_forward_validation(X: np.ndarray, Y: np.ndarray, train_n: int, test_n: int, **kwargs) -> float:
    metric_lst = []
    i = 0
    while True:
        X_train = X[i * test_n: train_n + i * test_n, :]
        Y_train = Y[i * test_n: train_n + i * test_n, :]
        X_test = X[train_n + i * test_n: train_n + (i + 1) * test_n, :]
        Y_test = Y[train_n + i * test_n: train_n + (i + 1) * test_n, :]

        rgsr = RandomForestRegressor(**kwargs)
        rgsr.fit(X_train, Y_train)
        Y_test_pred = rgsr.predict(X_test)
        metric_lst.append(error(Y_test.flatten(), Y_test_pred.flatten()))

        if train_n + (i + 1) * test_n >= X.shape[0]:
            break

        i += 1

    return np.mean(metric_lst)


if __name__ == '__main__':
    from src.air_passengers import load_x_series, transform_to_supervised

    # ---- 载入数据 ---------------------------------------------------------------------------------

    x_series = load_x_series()

    # ---- 转换为有监督样本集 ------------------------------------------------------------------------

    x_dim = 20  # 包括当前时刻为止的历史序列X维数
    y_dim = 10  # 模型预测序列y长度

    X, Y = transform_to_supervised(x_series, x_dim, y_dim)

    # ---- 确定每轮训练和测试样本数 ------------------------------------------------------------------
    # 这一步评估每次walk-forward时应该使用多少训练样本获得的模型指标最优.
    # 从最后结果可以看出, train_n = 15最合适.

    test_n = 5  # 手动设置每轮测试的样本数

    metric_train_n_lst = []
    for train_n in range(5, 60, 5):
        metric = walk_forward_validation(
            X, Y, train_n, test_n, n_estimators=100)
        metric_train_n_lst.append([train_n, metric])
        print('train_n: {}, metric: {}'.format(train_n, metric))
    proj_plt.plot([p[0] for p in metric_train_n_lst], [p[1]
                                                       for p in metric_train_n_lst])

    # ---- 确定训练模型参数 --------------------------------------------------------------------------
    # 这一步确定随机森林模型中的n_estimators参数.
    # 选择n_estimators = 150合适.

    train_n = 15

    repeat_n = 5
    metric_model_lst = []
    for n_estimators in range(10, 400, 50):
        metric_lst = []
        for i in range(repeat_n):
            metric = walk_forward_validation(
                X, Y, train_n, test_n, n_estimators=n_estimators)
            metric_lst.append(metric)
        metric_model = np.mean(metric_lst)
        metric_model_lst.append([n_estimators, metric_model])
        print('n_estimators: {}, mean MAE: {}'.format(
            n_estimators, metric_model))
    proj_plt.plot([p[0] for p in metric_model_lst], [p[1]
                                                     for p in metric_model_lst])

    # ---- 模型训练和验证 ---------------------------------------------------------------------------

    n_estimators = 150
    rgsr = RandomForestRegressor(n_estimators=n_estimators)

    X_train = X[-(train_n + test_n): -test_n, :]
    Y_train = Y[-(train_n + test_n): -test_n, :]
    X_test = X[-test_n:, :]

    rgsr.fit(X_train, Y_train)
    Y_test_pred = rgsr.predict(X_test)

    proj_plt.plot(Y[-3, :].flatten())
    proj_plt.plot(Y_test_pred[-3, :].flatten())
