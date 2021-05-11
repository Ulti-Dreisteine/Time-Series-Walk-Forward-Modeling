# -*- coding: utf-8 -*-
"""
Created on 2021/05/11 11:43:32

@File -> gbdt_walk_forward.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: GBDT Walk-forward模型验证
"""

import warnings
from matplotlib.pyplot import xcorr
from scipy.sparse import data

# warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error as error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(
    os.path.abspath(__file__), '../../../'))
sys.path.append(BASE_DIR)

from src.settings import *


def build_datasets(X: np.ndarray, Y: np.ndarray, train_n: int, test_n: int) -> dict:
    datasets = {}
    i = 0
    while True:
        X_train = X[i * test_n: train_n + i * test_n, :]
        Y_train = Y[i * test_n: train_n + i * test_n, :]
        # X_train = X[: train_n + i * test_n, :]
        # Y_train = Y[: train_n + i * test_n, :]
        X_test = X[train_n + i * test_n: train_n + (i + 1) * test_n, :]
        Y_test = Y[train_n + i * test_n: train_n + (i + 1) * test_n, :]

        if train_n + (i + 1) * test_n >= X.shape[0]:
            break

        datasets[i] = {
            'X_train': X_train,
            'Y_train': Y_train,
            'X_test': X_test,
            'Y_test': Y_test,
        }

        i += 1
    return datasets


if __name__ == '__main__':
    from src.air_passengers import load_x_series, transform_to_supervised

    # ---- 载入数据 ---------------------------------------------------------------------------------

    x_series = load_x_series()

    # ---- 转换为有监督样本集 ------------------------------------------------------------------------

    x_dim = 10  # 包括当前时刻为止的历史序列X维数
    y_dim = 10  # 模型预测序列y长度

    X, Y = transform_to_supervised(x_series, x_dim, y_dim)

    # ---- 归一化 -----------------------------------------------------------------------------------

    scaler_x = MinMaxScaler(feature_range=(0, 1))
    X = scaler_x.fit_transform(X)
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    Y = scaler_y.fit_transform(Y)

    # # ---- 模型训练 ---------------------------------------------------------------------------------

    train_n, test_n = 20, 10

    # 划分Walk-forward数据集.
    datasets = build_datasets(X, Y, train_n, test_n)

    # 训练模型.
    # gbdt = GradientBoostingRegressor(n_estimators=200, learning_rate = 0.1)
    # rgsr = MultiOutputRegressor(gbdt)
    # rgsr = RandomForestRegressor(n_estimators=200)

    # 一定要设置为warm_start = True, max_iter = 1, 否则每轮都会充分训练
    rgsr = MLPRegressor(
        solver='adam',
        hidden_layer_sizes=(10,),
        activation='tanh',
        learning_rate='adaptive',
        learning_rate_init=0.01,
        max_iter=1,
        alpha=0.001,
        warm_start=True
    )

    loss_lst = []
    for turn in range(30):
        metric_turn_lst = []
        for k in datasets.keys():
            X_train = datasets[k]['X_train']
            Y_train = datasets[k]['Y_train']
            X_test = datasets[k]['X_test']
            Y_test = datasets[k]['Y_test']

            rgsr.fit(X_train, Y_train)

            Y_test_pred = rgsr.predict(X_test)
            # Y_train_pred = rgsr.predict(X_train)

            loss = error(Y_test.flatten(), Y_test_pred.flatten())
            # loss = error(Y_train.flatten(), Y_train_pred.flatten())

            metric_turn_lst.append(loss)
        loss_lst.append(np.mean(metric_turn_lst))
        print('turn: {},\t metric: {}'.format(turn, np.mean(metric_turn_lst)))

    proj_plt.figure()
    proj_plt.plot(loss_lst)

# ---- 模型预测 -------------------------------------------------------------------------------------

x = x_series[-x_dim:].reshape(1, -1)
y_pred = rgsr.predict(x)
y_pred = scaler_y.inverse_transform(y_pred)

proj_plt.figure()
proj_plt.plot(
    np.hstack((x_series, y_pred.flatten())),
)
