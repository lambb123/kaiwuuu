#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np
from collections import deque

MAP_SIZE = 128.0

def _norm(v, v_max, v_min=0.0):
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0

def _ensure_list(data):
    """解决开悟平台底层 C++ 数据序列化可能退化为字典的 Bug"""
    if not data: return []
    if isinstance(data, list): return data
    if isinstance(data, dict):
        if "pos" in data or "hero_id" in data: return [data]
        return list(data.values())
    return [data]

class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 1000
        self.last_treasure_count = 0 
        self.last_nearest_t_dist = MAP_SIZE * 1.41
        self.last_buff_time = 0
        self.pos_history = deque(maxlen=11) # 追踪10步前位置，预判怪物2

    def feature_process(self, env_obs, last_action):
        observation = env_obs.get("observation", {})
        frame_state = observation.get("frame_state", {})
        env_info = observation.get("env_info", {})
        legal_act_raw = observation.get("legal_action", [])
        
        self.step_no = observation.get("step_no", 0)
        self.max_step = env_info.get("max_step", 1000)

        heroes_list = _ensure_list(frame_state.get("heroes"))
        hero = heroes_list[0] if heroes_list else {}
        hero_pos = hero.get("pos", {"x": 0, "z": 0})
        hx, hz = hero_pos.get("x", 0), hero_pos.get("z", 0)
        self.pos_history.append((hx, hz))

        # 1. 怪物2背刺预判特征
        if len(self.pos_history) == 11:
            h10_x, h10_z = self.pos_history[0]
            h10_dx_norm, h10_dz_norm = _norm(hx-h10_x, MAP_SIZE, -MAP_SIZE), _norm(hz-h10_z, MAP_SIZE, -MAP_SIZE)
        else:
            h10_dx_norm, h10_dz_norm = 0.5, 0.5

        # 2. 全局引力雷达（宝箱、怪物）
        nearest_t_dist = MAP_SIZE * 1.41
        t_dx_norm, t_dz_norm = 0.0, 0.0
        organs_list = _ensure_list(frame_state.get("organs"))
        for organ in organs_list:
            if organ.get("status") == 1 and organ.get("sub_type") == 1:
                ox, oz = organ.get("pos",{}).get("x",0), organ.get("pos",{}).get("z",0)
                dist = np.sqrt((ox-hx)**2 + (oz-hz)**2)
                if dist < nearest_t_dist:
                    nearest_t_dist = dist
                    t_dx_norm, t_dz_norm = np.clip((ox-hx)/MAP_SIZE, -1, 1), np.clip((oz-hz)/MAP_SIZE, -1, 1)
                    
        nearest_m_dist = MAP_SIZE * 1.41
        m_dx_norm, m_dz_norm = 0.0, 0.0
        monsters_list = _ensure_list(frame_state.get("monsters"))
        for m in monsters_list:
            mx, mz = m.get("pos",{}).get("x",0), m.get("pos",{}).get("z",0)
            dist = np.sqrt((mx-hx)**2 + (mz-hz)**2)
            if dist < nearest_m_dist:
                nearest_m_dist = dist
                m_dx_norm, m_dz_norm = np.clip((mx-hx)/MAP_SIZE, -1, 1), np.clip((mz-hz)/MAP_SIZE, -1, 1)

        # 3. 向量特征组装
        vector_feat = np.array([
            _norm(hx, MAP_SIZE), _norm(hz, MAP_SIZE),
            _norm(hero.get("flash_cooldown", 0), 2000.0),
            _norm(hero.get("buff_remaining_time", 0), 50.0),
            _norm(self.step_no, self.max_step), _norm(self.step_no, self.max_step),
            t_dx_norm, t_dz_norm, m_dx_norm, m_dz_norm,
            h10_dx_norm, h10_dz_norm
        ], dtype=np.float32)

        # 4. 视觉矩阵
        spatial_tensor = np.zeros((4, 21, 21), dtype=np.float32)
        map_info = observation.get("map_info", [])
        if map_info and len(map_info) == 21: spatial_tensor[0] = np.array(map_info, dtype=np.float32)
        for organ in organs_list:
            if organ.get("status") == 1:
                gx, gz = 10+(organ.get("pos",{}).get("x",0)-hx), 10+(organ.get("pos",{}).get("z",0)-hz)
                if 0<=gx<21 and 0<=gz<21: spatial_tensor[1 if organ.get("sub_type")==1 else 2, int(gx), int(gz)] = 1.0
        for m in monsters_list:
            if m.get("is_in_view", 0):
                gx, gz = np.clip(10+(m.get("pos",{}).get("x",0)-hx), 0, 20), np.clip(10+(m.get("pos",{}).get("z",0)-hz), 0, 20)
                spatial_tensor[3, int(gx), int(gz)] = 1.0

        feature = np.concatenate([spatial_tensor.flatten(), vector_feat])

        # 合法动作掩码 (16维)
        legal_action = [1]*16
        if isinstance(legal_act_raw, list) and len(legal_act_raw) >= 8:
            legal_action = [int(x) for x in legal_act_raw][:16]
            if len(legal_action) == 8: legal_action += [1]*8

        # ==========================================
        # 5. 奖励函数：针对你的监控图进行“处方级”调优
        # ==========================================
        reward = 0.01 + (0.02 * (self.step_no / self.max_step))
        
        # 指数级恐惧：靠近怪物重罚，逼出闪现！
        if nearest_m_dist < 5.0:
            reward -= np.clip(1.5 * (1.0 / (nearest_m_dist + 0.1)**2), 0, 3.5)
        elif nearest_m_dist < 8.0:
            reward -= 0.1

        # 宝箱引力：解决你图中“宝箱收集为0”
        cur_treasure = hero.get("treasure_collected_count", 0)
        if cur_treasure > self.last_treasure_count:
            reward += 5.0  # 巨额即时奖励
            self.last_treasure_count = cur_treasure
        else:
            if nearest_t_dist != MAP_SIZE * 1.41:
                # 持续引力场奖励
                reward += 0.4 * (1.0 / (nearest_t_dist / 10.0 + 1.0))
                if nearest_t_dist < self.last_nearest_t_dist: reward += 0.1

        # 怪物2出生杀防御
        if 270 < self.step_no < 330 and len(self.pos_history) == 11:
            dist_to_h10 = np.sqrt((hx-self.pos_history[0][0])**2 + (hz-self.pos_history[0][1])**2)
            if dist_to_h10 < 7.0: reward -= 0.8

        # 加速Buff拾取奖励（解决你图中加速Buff为0）
        cur_buff_time = hero.get("buff_remaining_time", 0)
        if cur_buff_time > 0 and self.last_buff_time == 0: reward += 1.5
        self.last_buff_time = cur_buff_time

        self.last_nearest_t_dist = nearest_t_dist
        return feature, legal_action, [reward]