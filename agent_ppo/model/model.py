#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
from agent_ppo.conf.conf import Config

def make_fc_layer(in_features, out_features):
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc

class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device
        # 视觉流 (CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(Config.SPATIAL_CHANNELS, 16, 3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2), # 10x10
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2), # 5x5
            nn.Flatten()
        )
        # 向量流 (MLP)
        self.mlp = nn.Sequential(
            make_fc_layer(Config.VECTOR_LEN, 64),
            nn.ReLU(),
            make_fc_layer(64, 64),
            nn.ReLU()
        )
        # 融合决策
        self.fusion = nn.Sequential(
            make_fc_layer(32*5*5 + 64, 512),
            nn.ReLU(),
            make_fc_layer(512, 256),
            nn.ReLU()
        )
        self.actor_head = make_fc_layer(256, Config.ACTION_NUM)
        self.critic_head = make_fc_layer(256, Config.VALUE_NUM)

    def forward(self, obs, inference=False):
        spatial_flat, vector_feat = obs[:, :Config.SPATIAL_LEN], obs[:, Config.SPATIAL_LEN:]
        spatial_tensor = spatial_flat.view(-1, Config.SPATIAL_CHANNELS, Config.MAP_SIZE, Config.MAP_SIZE)
        cnn_out = self.cnn(spatial_tensor)
        mlp_out = self.mlp(vector_feat)
        hidden = self.fusion(torch.cat([cnn_out, mlp_out], dim=1))
        return self.actor_head(hidden), self.critic_head(hidden)

    def set_train_mode(self): self.train()
    def set_eval_mode(self): self.eval()