# -*- coding: utf-8 -*-
"""
Created on 2021/05/11 11:43:58

@File -> __init__.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 初始化
"""

from scipy.ndimage.interpolation import shift
from typing import Tuple
import pandas as pd
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(
    os.path.abspath(__file__), '../../../'))
sys.path.append(BASE_DIR)

from src.settings import *

__all__ = ['load_x_series', 'transform_to_supervised']


def load_x_series() -> np.ndarray:
    data = pd.read_excel(os.path.join(
        PROJ_DIR, 'data/raw/air_passengers.xlsx'))
    x_series = data['value'].values.flatten()
    return x_series


def transform_to_supervised(x_series: np.ndarray, x_dim: int, y_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """将数据集转为适应有监督的模式

    :param x_series: x的时间序列
    :param x_dim: x样本维数
    :param y_dim: y样本维数
    :return: X和Y的数据集
    """
    x_series = x_series.copy().reshape(-1, 1)  # type: np.ndarray
    total_dim = x_dim + y_dim
    A = None
    for i in range(total_dim):
        x_shift = shift(x_series.copy(), shift=[-i, 0])

        if A is None:
            A = x_shift
        else:
            A = np.hstack((A, x_shift))

    A = A[: -(x_dim + y_dim - 1), :]
    X, Y = A[:, :x_dim], A[:, x_dim:]
    return X, Y
