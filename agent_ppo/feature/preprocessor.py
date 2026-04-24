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
        self.last_nearest_b_dist = MAP_SIZE * 1.41
        self.last_buff_time = 0
        self.last_flash_cd = 0
        self.pos_history = deque(maxlen=11)

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

        # 【防撞墙检测】：判断当前坐标是否与上一步完全一致
        is_stuck = False
        if len(self.pos_history) >= 2:
            last_hx, last_hz = self.pos_history[-2]
            if hx == last_hx and hz == last_hz:
                is_stuck = True

        # 1. 怪物2背刺预判特征
        if len(self.pos_history) == 11:
            h10_x, h10_z = self.pos_history[0]
            h10_dx_norm, h10_dz_norm = _norm(hx-h10_x, MAP_SIZE, -MAP_SIZE), _norm(hz-h10_z, MAP_SIZE, -MAP_SIZE)
        else:
            h10_dx_norm, h10_dz_norm = 0.5, 0.5

        # 2. 全局引力雷达（宝箱、Buff、怪物）
        nearest_t_dist = MAP_SIZE * 1.41
        nearest_b_dist = MAP_SIZE * 1.41
        t_dx_norm, t_dz_norm = 0.0, 0.0
        
        organs_list = _ensure_list(frame_state.get("organs"))
        for organ in organs_list:
            if organ.get("status") == 1:
                sub_type = organ.get("sub_type")
                ox, oz = organ.get("pos",{}).get("x",0), organ.get("pos",{}).get("z",0)
                dist = np.sqrt((ox-hx)**2 + (oz-hz)**2)
                
                if sub_type == 1:  # 宝箱
                    if dist < nearest_t_dist:
                        nearest_t_dist = dist
                        t_dx_norm, t_dz_norm = np.clip((ox-hx)/MAP_SIZE, -1, 1), np.clip((oz-hz)/MAP_SIZE, -1, 1)
                elif sub_type == 2:  # 加速Buff
                    if dist < nearest_b_dist:
                        nearest_b_dist = dist
                    
        nearest_m_dist = MAP_SIZE * 1.41
        m_dx_norm, m_dz_norm = 0.0, 0.0
        monsters_list = _ensure_list(frame_state.get("monsters"))
        for m in monsters_list:
            mx, mz = m.get("pos",{}).get("x",0), m.get("pos",{}).get("z",0)
            dist = np.sqrt((mx-hx)**2 + (mz-hz)**2)
            if dist < nearest_m_dist:
                nearest_m_dist = dist
                m_dx_norm, m_dz_norm = np.clip((mx-hx)/MAP_SIZE, -1, 1), np.clip((mz-hz)/MAP_SIZE, -1, 1)

        # 闪现最大冷却可配置到2000，这里保持2000.0做归一化是安全的
        flash_cd = hero.get("flash_cooldown", 0)

        # 3. 向量特征组装
        vector_feat = np.array([
            _norm(hx, MAP_SIZE), _norm(hz, MAP_SIZE),
            _norm(flash_cd, 2000.0),
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

        # 合法动作掩码 (16维)：精确屏蔽闪现CD
        legal_action = [1] * 16
        if isinstance(legal_act_raw, list):
            if len(legal_act_raw) >= 16:
                legal_action = [int(x) for x in legal_act_raw][:16]
            elif len(legal_act_raw) == 8:
                legal_action[:8] = [int(x) for x in legal_act_raw]
                if flash_cd > 0:
                    legal_action[8:16] = [0] * 8

        # ==========================================
        # 5. 奖励函数：融合高端操作的精细化调优
        # ==========================================
        # 基础生存奖励
        reward = 0.015 
        
        # 撞墙/原地不动惩罚：抵消生存分，逼迫其绕开障碍物
        if is_stuck:
            reward -= 0.1 

        # 【吃宝箱】：加入了"安全锁"逻辑
        cur_treasure = hero.get("treasure_collected_count", 0)
        if cur_treasure > self.last_treasure_count:
            reward += 1.5  
            self.last_treasure_count = cur_treasure
        else:
            # 只有在怪物距离较远（安全）时，才允许产生宝箱引力
            if nearest_t_dist != MAP_SIZE * 1.41 and nearest_m_dist > 5.0:
                reward += 0.02 * (1.0 / (nearest_t_dist / 10.0 + 1.0))
                if nearest_t_dist < self.last_nearest_t_dist: 
                    reward += 0.01

        # 【吃加速】：适配新的怪物加速时间（300步）
        cur_buff_time = hero.get("buff_remaining_time", 0)
        if cur_buff_time > 0 and self.last_buff_time == 0: 
            reward += 1.0  
        elif cur_buff_time == 0 and self.step_no > 200:
            # 【修改点】由于怪物300步就加速，所以这里改为200步后就要开始强制寻找Buff
            if nearest_b_dist != MAP_SIZE * 1.41 and nearest_m_dist > 4.0:
                reward += 0.03 * (1.0 / (nearest_b_dist / 10.0 + 1.0))
                if nearest_b_dist < self.last_nearest_b_dist:
                    reward += 0.015
        
        # 【交闪现】：极限救场 vs 乱交技能
        if flash_cd > self.last_flash_cd: 
            if nearest_m_dist < 6.0:
                reward += 0.5  
            else:
                reward -= 0.2  

        # 怪物极度恐惧惩罚
        if nearest_m_dist <= 2.0:
            reward -= 1.5  
        elif nearest_m_dist < 6.0:
            reward -= 0.2 * (1.0 / (nearest_m_dist + 0.1))

        # 【防背刺】：适配新的怪物2出生时间（200步）
        # 【修改点】第200步在10步前的位置生成怪物，所以在185~215步期间强行逼迫其离开10步前的位置
        if 185 < self.step_no < 215 and len(self.pos_history) == 11:
            dist_to_h10 = np.sqrt((hx-self.pos_history[0][0])**2 + (hz-self.pos_history[0][1])**2)
            if dist_to_h10 < 6.0: 
                reward -= 0.5

        # 状态留存
        self.last_nearest_t_dist = nearest_t_dist
        self.last_nearest_b_dist = nearest_b_dist
        self.last_buff_time = cur_buff_time
        self.last_flash_cd = flash_cd

        return feature, legal_action, [reward]