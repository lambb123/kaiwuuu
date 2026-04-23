#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

class Config:
    # 空间特征：4通道 x 21 x 21 (地形、宝箱、Buff、怪物)
    SPATIAL_CHANNELS = 4
    MAP_SIZE = 21
    SPATIAL_LEN = SPATIAL_CHANNELS * MAP_SIZE * MAP_SIZE  # 1764

    # 全局向量特征：基础(6) + 增强雷达(4) + 历史轨迹预判(2) = 12维
    VECTOR_LEN = 12 

    # 总输入维度
    DIM_OF_OBSERVATION = SPATIAL_LEN + VECTOR_LEN

    # 动作空间：16 (8个方向移动 + 8个方向闪现)
    ACTION_NUM = 16
    VALUE_NUM = 1

    # PPO 超参数调优
    GAMMA = 0.995     # 让鲁班更“高瞻远瞩”，为了活过1000步可以放弃眼前的小利
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.0003
    BETA_START = 0.01  # 保持高探索，解决你图中“闪现次数为0”的问题
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    GRAD_CLIP_RANGE = 0.5