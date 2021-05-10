# -*- coding: utf-8 -*-
"""
Created on 2021/05/10 14:46:53

@File -> walk_forward_modeling.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 飞机乘客数据建模
"""

from sklearn.ensemble import RandomForestRegressor
from scipy.ndimage.interpolation import shift
from sklearn.metrics import mean_squared_error as error
from typing import Tuple
import pandas as pd
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '../../../'))
sys.path.append(BASE_DIR)

from src.settings import *


def gen_total_datasets(x_series: np.ndarray, hist_len: int, pred_len: int) -> Tuple[np.ndarray, np.ndarray]:
    x_series = x_series.copy().reshape(-1, 1)  # type: np.ndarray
    total_len = hist_len + pred_len

    A = None
    for i in range(total_len):
        x_shift = shift(x_series.copy(), shift = [-i, 0])

        if A is None:
            A = x_shift
        else:
            A = np.hstack((A, x_shift))

    A = A[: -(hist_len + pred_len - 1), :]
    X, Y = A[:, :hist_len], A[:, hist_len:]
    return X, Y


def walk_forward_validation(X: np.ndarray, Y: np.ndarray, train_n: int, test_n: int, **kwargs) -> float:
    MAE_lst = []
    i = 0
    while True:
        X_train = X[i * test_n : train_n + i * test_n, :]
        Y_train = Y[i * test_n : train_n + i * test_n, :]
        X_test = X[train_n + i * test_n : train_n + (i + 1) * test_n, :]
        Y_test = Y[train_n + i * test_n : train_n + (i + 1) * test_n, :]

        rgsr = RandomForestRegressor(**kwargs)
        rgsr.fit(X_train, Y_train)
        Y_test_pred = rgsr.predict(X_test)
        MAE_lst.append(error(Y_test.flatten(), Y_test_pred.flatten()))

        # proj_plt.figure()
        # proj_plt.plot(Y_test.flatten())
        # proj_plt.plot(Y_test_pred.flatten())
        # proj_plt.show()

        if train_n + (i + 1) * test_n >= X.shape[0]:
            break

        i += 1
    
    return np.mean(MAE_lst)


if __name__ == '__main__':
    
    # ---- 载入数据 ---------------------------------------------------------------------------------

    data = pd.read_excel(os.path.join(PROJ_DIR, 'data/raw/air_passengers.xlsx'))
    # proj_plt.plot(data['value'])

    # ---- 构造总体训练集和测试集 --------------------------------------------------------------------

    HIST_LEN = 20  # 包括当前时刻为止的历史序列X长度
    PRED_LEN = 20  # 模型预测序列y长度

    x_series = data['value'].values.reshape(-1, 1)
    X, Y = gen_total_datasets(x_series, HIST_LEN, PRED_LEN)
    
    test_n = 1

    # ---- 使用Walk-forward方法训练 -----------------------------------------------------------------
    # 这一步评估每次walk-forward时应该使用多少训练样本获得的模型指标最优.

    # eval_results = []
    # for train_n in range(5, 60, 5):
    #     MAE_mean = walk_forward_validation(X, Y, train_n, test_n, n_estimators=200)
    #     eval_results.append([train_n, MAE_mean])
    #     print('train_n: {}, mean MAE: {}'.format(train_n, MAE_mean))
    # proj_plt.plot([p[0] for p in eval_results], [p[1] for p in eval_results])

    # ---- Walk-forward超参数确定 -------------------------------------------------------------------

    # train_n = 15

    # eval_results = []
    # repeat_n = 1
    # for n_estimators in range(10, 400, 50):
    #     MAE_lst = []
    #     for i in range(repeat_n):
    #         MAE_mean = walk_forward_validation(X, Y, train_n, test_n, n_estimators=n_estimators)
    #         MAE_lst.append(MAE_mean)
    #     MAE_value = np.mean(MAE_lst)
    #     eval_results.append([n_estimators, MAE_value])
    #     print('n_estimators: {}, mean MAE: {}'.format(n_estimators, MAE_value))
    # proj_plt.plot([p[0] for p in eval_results], [p[1] for p in eval_results])

    # ---- 模型训练和验证 ---------------------------------------------------------------------------

    train_n = 15
    rgsr = RandomForestRegressor(n_estimators=200)

    X_train = X[-(train_n + test_n) : -test_n, :]
    Y_train = Y[-(train_n + test_n) : -test_n, :]
    # X_train = X[:-1, :]
    # Y_train = Y[:-1, :]
    X_test = X[-1:, :]
    rgsr.fit(X_train, Y_train)
    Y_test_pred = rgsr.predict(X_test)
    # proj_plt.plot(
    #     np.hstack((x_series.flatten(), Y_test_pred.flatten()))
    #     )
    proj_plt.plot(Y[-1, :].flatten())
    proj_plt.plot(Y_test_pred.flatten())