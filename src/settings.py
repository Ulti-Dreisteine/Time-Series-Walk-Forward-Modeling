# -*- coding: utf-8 -*-
"""
Created on 2020/1/21 下午3:01

@Project -> File: pollution-forecast-offline-training-version-2 -> settings.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 默认设置
"""

import sys

sys.path.append('../')

from mod.config.config_loader import config_loader

PROJ_DIR, PROJ_CMAP = config_loader.proj_dir, config_loader.proj_cmap
proj_plt = config_loader.proj_plt

# 载入项目变量配置.
ENC_CONFIG = config_loader.environ_config
MODEL_CONFIG = config_loader.model_config
TEST_PARAMS = config_loader.test_params

# ---- 定义环境变量 ---------------------------------------------------------------------------------

# ---- 定义模型参数 ---------------------------------------------------------------------------------

# ---- 定义测试参数 ---------------------------------------------------------------------------------

# ---- 定义通用函数 ---------------------------------------------------------------------------------
