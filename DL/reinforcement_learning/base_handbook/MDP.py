#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   MDP.py
@Time   :   2024/07/19 11:22:26
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   学习马尔科夫决策过程
            https://hrl.boyuai.com/chapter/1/%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E5%86%B3%E7%AD%96%E8%BF%87%E7%A8%8B#321-%E9%9A%8F%E6%9C%BA%E8%BF%87%E7%A8%8B
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')
import numpy as np
np.random.seed(0)

# ================================================================
# =================== 简单MRP Markov reward process ==============
# ================================================================
# 定义状态转移概率矩阵P
Ptrs = [
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]
Ptrs = np.array(Ptrs)
rewards = [-1, -2, -2, 10, 1, 0]  # 定义奖励函数，此处直接返回固定值，实际上是r(s)的函数
gamma = 0.5  # 定义折扣因子


def compute_return(start_index, chain, gamma):
    """给定一条序列，计算从某个索引（起始状态）到序列终点（终止状态）得到的回报（return）
    
    Args:
        chain: 采样的结果，即序列episode
    """
    G = 0
    for i in reversed(range(start_index, len(chain))):
        sth = rewards[chain[i] - 1]
        print(f"i = {i}, 当前即时奖励为： {sth}")
        # 从后向前累计幂次，本质是减少gamma的重复计算的技巧，斐波那契
        G = gamma * G + sth
    return G


"""
奖励、回报与价值：
    其中奖励和价值是针对当前状态的，回报是针对一个采样序列
    奖励：
        转移至当前状态获取的奖励
    回报：
        整个状态序列，加上衰减加权后奖励的总和
        如果要跟当前状态扯上关系，则是：当前状态 + 某一条确定的采样序列
    价值：
        当前状态在转移至下一个状态时，有很多可能状态选择
        可以理解为：当前状态 + 所有可能的采样序列
        价值则是，这种所有可能采样的期望
    价值函数：
        所有状态的价值就形成了价值函数
        价值函数的输入为某个状态，输出为这个状态的价值
"""
def compute(P, rewards, gamma, states_num):
    """利用贝尔曼方程的矩阵形式，求解价值函数的解析解，即每个状态的价值
    
    Args:
        states_num: MRP的状态数，用来构建单位矩阵
    """
    rewards = np.array(rewards).reshape((-1, 1))  # 向量
    value = np.dot(
        np.linalg.inv(np.eye(states_num, states_num) - gamma * P),
        rewards
    )
    return value


# ================================================================
# ==================== MDP Markov decision process ===============
# ================================================================
"""
新引入：
    动作：
        动作a会对环境产生影响，即影响奖励函数与转移矩阵
    策略：
        π(a|s)
被修改：
    奖励函数：
        r(s)  -->  r(s, a)
    状态转移矩阵：
        P(s'|s)  -->  P(s'|s, a)
"""
S = ["s1", "s2", "s3", "s4", "s5"]  # 状态集合
A = ["保持s1", "前往s1", "前往s2", "前往s3", "前往s4", "前往s5", "概率前往"]  # 动作集合
# 状态转移函数
P = {
    "s1-保持s1-s1": 1.0,
    "s1-前往s2-s2": 1.0,
    "s2-前往s1-s1": 1.0,
    "s2-前往s3-s3": 1.0,
    "s3-前往s4-s4": 1.0,
    "s3-前往s5-s5": 1.0,
    "s4-前往s5-s5": 1.0,
    "s4-概率前往-s2": 0.2,
    "s4-概率前往-s3": 0.4,
    "s4-概率前往-s4": 0.4,
}
# 奖励函数
R = {
    "s1-保持s1": -1,
    "s1-前往s2": 0,
    "s2-前往s1": -1,
    "s2-前往s3": -2,
    "s3-前往s4": -2,
    "s3-前往s5": 0,
    "s4-前往s5": 10,
    "s4-概率前往": 1,
}
gamma = 0.5  # 折扣因子
MDP = (S, A, P, R, gamma)

# 策略1,随机策略
Pi_1 = {
    "s1-保持s1": 0.5,
    "s1-前往s2": 0.5,
    "s2-前往s1": 0.5,
    "s2-前往s3": 0.5,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.5,
    "s4-概率前往": 0.5,
}
# 策略2
Pi_2 = {
    "s1-保持s1": 0.6,
    "s1-前往s2": 0.4,
    "s2-前往s1": 0.3,
    "s2-前往s3": 0.7,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.1,
    "s4-概率前往": 0.9,
}

# 把输入的两个字符串通过“-”连接,便于使用上述定义的P、R变量
def join(str1, str2):
    return str1 + '-' + str2


"""
解析解解法：
    边缘化： 将a在其取值集合A的维度上进行聚合
            对于某一个状态，我们根据策略所有动作的概率进行加权，得到的奖励和就可以认为是一个 MRP 在该状态下的奖励

    公式：
        见链接
    
    特点：
        不适合动作空间A非常大情景
"""
gamma = 0.5
# 转化后的MRP的状态转移矩阵
P_from_mdp_to_mrp = [
    [0.5, 0.5, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.5, 0.5],
    [0.0, 0.1, 0.2, 0.2, 0.5],
    [0.0, 0.0, 0.0, 0.0, 1.0],
]
P_from_mdp_to_mrp = np.array(P_from_mdp_to_mrp)
R_from_mdp_to_mrp = [-0.5, -1.5, -1.0, 5.5, 0]


# ================================================================
# ============================ 蒙特卡洛近似 ========================
# ================================================================
"""
近似：
    通过统计特征来估算目标量（状态价值）
"""
def sample(MDP, Pi, timestep_max, number):
    """采样序列
    
    Args:
        Pi: 策略
    """
    S, A, P, R, gamma = MDP
    episodes = []
    for _ in range(number):
        episode = []
        timestep = 0
        s = S[np.random.randint(4)]  # 随机选择除了s5之外的状态作为起始
        while s != "s5" and timestep <= timestep_max:
            timestep += 1
            rand, temp = np.random.rand(), 0
            # 在状态s下根据策略选择动作
            for a_opt in A:
                temp += Pi.get(join(s, a_opt), 0)
                if temp > rand:
                    a = a_opt
                    r = R.get(join(s, a), 0)
                    break
            rand, temp = np.random.rand(), 0
            # 根据状态转移概率得到下一个状态s_next
            for s_opt in S:
                temp += P.get(join(join(s, a), s_opt), 0)
                if temp > rand:
                    s_next = s_opt
                    break
            episode.append((s, a, r, s_next))  # 把（s,a,r,s_next）元组放入序列中
            s = s_next  # s_next变成当前状态,开始接下来的循环
        episodes.append(episode)
    return episodes


def MC(episodes, V, N, gamma):
    """对所有序列计算状态的价值
    """
    for episode in episodes:
        G = 0
        for i in reversed(range(len(episode))):
            (s, a, r, s_next) = episode[i]
            G = gamma * G + r
            N[s] = N[s] + 1
            V[s] = V[s] + (G - V[s]) / N[s]


# ================================================================
# ============================= 占用度量 ==========================
# ================================================================
"""
引入概念：
    在上面的蒙特卡洛近似中，我们随机定义了初始状态，然后根据转移矩阵得到下一个时间步的状态
    当我们将这种行为进行抽象时，就能定义在时间步t下，因为策略π导致状态为s的概率：
        P^\pi_t(s)
    若定义初始状态为\nu_0(s)， 则\nu_0(s) = P^\pi_0(s)

    策略的访问概率分布：
        \nu^\pi(s) = (1-\gamma)\sum_{t=0}^{\inf}\gamma^tP^\pi_t(s)
        表示：
            一个策略和 MDP 交互会访问到的状态的分布
    
    策略的占用度量：
        \rho^\pi(s,a)=(1-\gamma)\sum_{t=0}^{\inf}\gamma^tP^\pi_t(s)\pi(a|s)
        表示：
    
    两个定理：
        1. 智能体使用不用策略π1、π2和同一个MDP交互，如果占用度量一致，则策略一致  
        2. 若生成了占用度量\rho，则生成该度量的策略也唯一  

    理解：
        占用即occupancy，可以理解为重合
        即：
            实际的概率（是我们无法计算出来的，超验的）为A
            观测的概率（能通过统计去近似的）为B
        那么：
            如果A = B，就是B完全占用了A，则理解为B在统计意义上等于A
"""
def occupancy(episodes, s, a, timestep_max, gamma):
    """通过近似估计，统计状态动作对(s, a)的频率，来估算状态动作对的概率
    以此来估算策略的占用度量
    方法：
        设置一个较大的最大采样步数，采样很多次来统计(s, a)概率
    """
    rho = 0
    total_times = np.zeros(timestep_max)
    occur_times = np.zeros(timestep_max)  # 记录(s_t,a_t)=(s,a)的次数
    for episode in episodes:
        for i in range(len(episode)):
            (s_opt, a_opt, r, s_next) = episode[i]
            total_times[i] += 1
            if s == s_opt and a == a_opt:
                occur_times[i] += 1
    for i in reversed(range(timestep_max)):
        if total_times[i]:
            # 本质上还是那么渐进式计算均值的公式
            rho += gamma**i * occur_times[i] / total_times[i]
    return (1 - gamma) * rho


if __name__ == "__main__":
    # chain = [1, 2, 3, 6]
    # start_index = 0
    # G = compute_return(start_index, chain, gamma)
    # print("根据本序列计算得到回报为：%s。" % G)
    # V = compute(Ptrs, rewards, gamma, 6)
    # print("MRP中每个状态价值分别为\n", V)
    # V = compute(P_from_mdp_to_mrp, R_from_mdp_to_mrp, gamma, 5)
    # print("MDP中每个状态价值分别为\n", V)

    # # 采样5次,每个序列最长不超过20步
    # episodes = sample(MDP, Pi_1, 20, 5)
    # print('第一条序列\n', episodes[0])
    # print('第二条序列\n', episodes[1])
    # print('第五条序列\n', episodes[4])

    # timestep_max = 20
    # # 采样1000次,可以自行修改
    # episodes = sample(MDP, Pi_1, timestep_max, 1000)
    # gamma = 0.5
    # V = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
    # N = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
    # MC(episodes, V, N, gamma)
    # print("使用蒙特卡洛方法计算MDP的状态价值为\n", V)

    gamma = 0.5
    timestep_max = 1000
    episodes_1 = sample(MDP, Pi_1, timestep_max, 1000)
    episodes_2 = sample(MDP, Pi_2, timestep_max, 1000)
    rho_1 = occupancy(episodes_1, "s4", "概率前往", timestep_max, gamma)
    rho_2 = occupancy(episodes_2, "s4", "概率前往", timestep_max, gamma)
    # 不同的策略对于同一个状态动作对的占用度量是不一样的
    # 可以理解为：观测的方法不一样，那么估算的结果就不一样
    print(rho_1, rho_2)