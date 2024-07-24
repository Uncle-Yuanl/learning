#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   DP.py
@Time   :   2024/07/22 14:59:46
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   使用动态规划算法解决悬崖漫步和冰湖问题
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')
import copy


class CliffWalkingEnv:
    """悬崖漫步问题
    矩阵，m*n，有一边是悬崖
    每走一步奖励-1，掉下悬崖奖励-100
    """
    def __init__(self, ncol=12, nrow=4) -> None:
        self.ncol = ncol
        self.nrow = nrow
        # 定义转移矩阵P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
        # dp算法中的状态转移方程，但这玩意怎么看起来像策略
        self.P = self.createP()

    def createP(self):
        """定义环境
        其实就是动作->环境->奖励
        """
        # 每个格子上下左右4种转移
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]
        # 4种动作, change[0]:上,change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        # []里的是，横纵坐标x, y下一步的位置，初始在[0, 0]那么[0, 1]就是y从0变1，就是向下走一步
        # change就是walk
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    # 结束状态：【已经】掉进悬崖或者走到终点
                    # 无法转移，p=1, r=0
                    if i == self.nrow - i and j > 0:
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0, True)]
                        continue
                    
                    # 其他位置
                    next_x = min(self.ncol -1, max(0, j + change[a][0]))
                    next_y = min(self.nrow -1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x  # 横纵坐标在展开的P中的下标
                    reward = -1
                    done = False
                    # 下个位置在悬崖或是终点
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol - 1:  # 下一个位置在悬崖
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P
    

class PolicyIteration:
    """策略迭代算法
    """
    def __init__(self, env: CliffWalkingEnv, theta, gamma) -> None:
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow  # # 初始化价值为0
        self.pi = [[0.25, 0.25, 0.25, 0.25]   # 初始化为均匀随机策略
                   for i in range(self.env.ncol * self.env.nrow)]
        self.theta = theta  # 策略评估收敛阈值
        self.gamma = gamma  # 折扣因子

    def policy_evaluation(self):
        """策略评估：计算一个策略的状态价值函数
        子问题：
            每个状态的价值函数
        状态价值函数：
            sum(动作价值函数qsa)
        动态规划：
            使用t + 1的状态价值函数更新当前t的状态价值
        """
        cnt = 1
        while 1:
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = []  # 开始计算状态s下所有Q(s,a)价值
                for a in range(4):
                    qsa = 0
                    # 这个for有点不懂，在env中故意加了一层元组
                    # 如果必定只有一个res，那么直接qsa = 就好，为啥要+=
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        # 用下一个状态的价值更新当前状态的动作价值
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                    # 本章环境比较特殊,奖励和下一个状态有关,所以需要和状态转移概率相乘
                    qsa_list.append(self.pi[s][a] * qsa)
                new_v[s] = sum(qsa_list)  # 状态价值函数和动作价值函数之间的关系
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))  # 不动点，即收敛了
            print(f"old value: {[round(x, 1) for x in self.v[:12]]}")
            print(f"new value: {[round(x, 1) for x in new_v[:12]]}")
            self.v = new_v
            if max_diff < self.theta:
                break  # 满足收敛条件,退出评估迭代
            else:
                print(f"第{cnt}轮策略评估, 价值差为：{max_diff}，收敛值为：{self.theta}")
            cnt += 1
        print("策略评估进行%d轮后完成" % cnt)

    def policy_improvement(self):
        """策略提升
        贪心地在每一个状态选择动作价值最大的动作
        """
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)  # 计算有几个动作得到了最大的Q值
            # 这些动作均分概率
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]
        print("策略提升完成")
        return self.pi
    
    def policy_iteration(self):
        """策略迭代
        """
        while 1:
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi)  # 将列表进行深拷贝,方便接下来进行比较
            new_pi = self.policy_improvement()
            if old_pi == new_pi:
                break

def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("状态价值：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 为了输出美观,保持输出6个字符
            print('%6.6s' % ('%.3f' % agent.v[i * agent.env.ncol + j]), end=' ')
        print()

    print("策略：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 一些特殊的状态,例如悬崖漫步中的悬崖
            if (i * agent.env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * agent.env.ncol + j) in end:  # 目标状态
                print('EEEE', end=' ')
            else:
                a = agent.pi[i * agent.env.ncol + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()


class ValueIteration:
    """价值迭代
    Motivation:
        在策略迭代中，策略评估需要迭代非常多轮，直到状态价值收敛
        可能出现这种情况：状态价值函数还没有收敛，但是不论接下来怎么更新状态价值，策略提升得到的都是同一个策略
    Method:
        策略评估中进行一轮价值更新，然后直接根据更新后的价值进行策略提升
    细节：
        价值迭代中不存在显式的策略，我们只维护一个状态价值函数
        公式：贝尔曼最优方程
        解出最优状态价值函数，然后解析出最优策略
    """
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow  # 初始化价值为0
        self.theta = theta  # 价值收敛阈值
        self.gamma = gamma
        # 价值迭代结束后得到的策略
        self.pi = [None for i in range(self.env.ncol * self.env.nrow)]

    def value_iteration(self):
        cnt = 0
        while 1:
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = []  # 开始计算状态s下的所有Q(s,a)价值
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                    # 这一行和下一行代码是价值迭代和策略迭代的主要区别
                    # 没有乘转移概率了
                    qsa_list.append(qsa)
                # 选择最大的而不是sum
                new_v[s] = max(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta:
                break  # 满足收敛条件,退出评估迭代
            cnt += 1
        print("价值迭代一共进行%d轮" % cnt)
        self.get_policy()

    def get_policy(self):
        """根据价值函数导出一个贪婪策略
        所以，value_iteration中才能直接new_v[s] = max(qsa_list)？？
        """
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)  # 计算有几个动作得到了最大的Q值
            # 让这些动作均分概率
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]


if __name__ == "__main__":
    env = CliffWalkingEnv()
    action_meaning = ['^', 'v', '<', '>']
    theta = 0.001
    gamma = 0.9

    # 策略迭代
    agent = PolicyIteration(env, theta, gamma)
    agent.policy_iteration()

    # # 价值迭代
    # agent = ValueIteration(env, theta, gamma)
    # agent.value_iteration()

    print_agent(agent, action_meaning, list(range(37, 47)), [47])