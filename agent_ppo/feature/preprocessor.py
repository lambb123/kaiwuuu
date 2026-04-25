#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np
from collections import deque

MAP_SIZE = 128.0

def _norm(v, v_max, v_min=0.0):
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0

def _ensure_list(data):
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

        # 1. 怪物2背刺预判特征
        if len(self.pos_history) == 11:
            h10_x, h10_z = self.pos_history[0]
            h10_dx_norm, h10_dz_norm = _norm(hx-h10_x, MAP_SIZE, -MAP_SIZE), _norm(hz-h10_z, MAP_SIZE, -MAP_SIZE)
        else:
            h10_dx_norm, h10_dz_norm = 0.5, 0.5

        # 2. 全局引力雷达
        nearest_t_dist = MAP_SIZE * 1.41
        nearest_b_dist = MAP_SIZE * 1.41
        t_dx_norm, t_dz_norm = 0.0, 0.0
        
        organs_list = _ensure_list(frame_state.get("organs"))
        for organ in organs_list:
            if organ.get("status") == 1:
                sub_type = organ.get("sub_type")
                ox, oz = organ.get("pos",{}).get("x",0), organ.get("pos",{}).get("z",0)
                dist = np.sqrt((ox-hx)**2 + (oz-hz)**2)
                
                if sub_type == 1:  
                    if dist < nearest_t_dist:
                        nearest_t_dist = dist
                        t_dx_norm, t_dz_norm = np.clip((ox-hx)/MAP_SIZE, -1, 1), np.clip((oz-hz)/MAP_SIZE, -1, 1)
                elif sub_type == 2:  
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

        flash_cd = hero.get("flash_cooldown", 0)

        # 3. 向量特征
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

        # 精确屏蔽闪现CD
        legal_action = [1] * 16
        if isinstance(legal_act_raw, list):
            if len(legal_act_raw) >= 16:
                legal_action = [int(x) for x in legal_act_raw][:16]
            elif len(legal_act_raw) == 8:
                legal_action[:8] = [int(x) for x in legal_act_raw]
                if flash_cd > 0:
                    legal_action[8:16] = [0] * 8

        # ==========================================
        # 5. 奖励函数：彻底打破局部最优，强制跑动与寻宝
        # ==========================================
        reward = 0.02  # 提高基础存活分
        
        # 【猛药1：十步防蹲坑】
        if len(self.pos_history) == 11:
            p_list = list(self.pos_history)
            xs = [p[0] for p in p_list]
            zs = [p[1] for p in p_list]
            # 如果过去10步的活动范围小于 2x2（在原地打转或卡墙），给予毁灭性惩罚
            if max(xs) - min(xs) <= 2.0 and max(zs) - min(zs) <= 2.0:
                reward -= 1.0
                
        # 【猛药2：加速Buff最高优先级】
        cur_buff_time = hero.get("buff_remaining_time", 0)
        if cur_buff_time > 0 and self.last_buff_time == 0: 
            reward += 10.0  # 重赏！吃Buff是活下去的唯一希望
        elif cur_buff_time == 0:
            # “面包屑”引诱：只要没Buff，每靠近Buff一步就给分
            if nearest_b_dist != MAP_SIZE * 1.41:
                if nearest_b_dist < self.last_nearest_b_dist:
                    reward += 0.05
                    
        # 【猛药3：安全贪婪吃宝箱】
        cur_treasure = hero.get("treasure_collected_count", 0)
        if cur_treasure > self.last_treasure_count:
            reward += 5.0
            self.last_treasure_count = cur_treasure
        else:
            # 只有当怪物距离远于5格时，才用面包屑引诱它去吃宝箱（绝对不硬吃）
            if nearest_m_dist > 5.0 and nearest_t_dist != MAP_SIZE * 1.41:
                if nearest_t_dist < self.last_nearest_t_dist: 
                    reward += 0.03

        # 【怪物与闪现管理】
        if nearest_m_dist < 4.0:
            # 距离越近，扣分越狠 (线性斜坡惩罚)
            reward -= 0.4 * (4.0 - nearest_m_dist)
            
        # 贴脸必死极度惩罚
        if nearest_m_dist <= 1.5:
            reward -= 2.0

        # 交闪现判定
        if flash_cd > self.last_flash_cd: 
            if nearest_m_dist < 4.5:
                reward += 2.0  # 极限逃生，好评！
            else:
                reward -= 1.0  # 乱交技能，重罚！

        # 【防背刺】: 第200步左右避开10步前的位置
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