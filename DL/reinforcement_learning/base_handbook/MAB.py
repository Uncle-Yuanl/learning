#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   MAB.py
@Time   :   2024/07/18 15:38:31
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   经典多臂老虎机（multi-armed bandit）问题与方案
            问题定义：
                老虎机有K个杆，每个杆都服从奖励分布R
                目的是：拉T次后，获取最高奖励
            注意：
                这个问题是环境不变的
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')
import numpy as np
import matplotlib.pyplot as plt


class BernoulliBandit:
    """伯努利多臂老虎机,输入K表示拉杆个数
    p的概率拉出1,1-p的概率拉出0
    """
    def __init__(self, K) -> None:
        # 设置每个拉杆的p
        self.probs = np.random.uniform(size=K)
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        self.K = K

    def step(self, k):
        """杆的获奖概率是一定的，设置为了p
        但是每次拉（action）与环境（老虎机）作用下产生的奖励也是一个概率
        并不是拉倒了就是p，而是以p的概率生成1
        """
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0


class Solver:
    """多臂老虎机算法基本框架
    包括：
        - 根据策略选择动作
        - 根据动作生成奖励
        - 更新期望奖励估值（理解是对第k个杆p的估计）
        - 更新累计懊悔和计数
    """
    def __init__(self, bandit: BernoulliBandit) -> None:
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)
        self.regret = 0.   # 记录当前步的累计懊悔
        self.actions = []  # 维护列表，记录每一步的动作
        self.regrets = []  # 维护每一步的累计懊悔

    def run_one_step(self):
        """返回当前动作选择哪一根杆，由具体策略决定
        """
        raise NotImplementedError

    def update_regret(self, k):
        """在策略产生动作后，执行动作，更新累计懊悔并保存
        由于是伯努利分布，奖励期望就是p
        """
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run(self, num_steps):
        """运行一定次数
        """
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)


def plot_results(solvers, solver_names):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()


# ============================================================================
# ============================= 设计策略 ======================================
# ============================================================================
class EpsilonGreedy(Solver):
    """改进贪心算法
    有epsilon的概率从非最优杆中选择
    """
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super().__init__(bandit)
        self.epsilon = epsilon
        # 对于每个杆奖励分布的观测（估计）
        self.estimates = np.array([init_prob] * self.bandit.K)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            # 纯随机
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)  # 选择期望奖励估值最大的拉杆

        # 执行动作，获取本次动作的奖励
        r = self.bandit.step(k)
        # 更新对k的估计
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


class DecayingEpsilonGreedy(Solver):
    """epsilon随时间递减，因为假设观测越多，概率估计越接近真实值
    """
    def __init__(self, bandit, init_prob=1.0):
        super().__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:  # epsilon值随时间衰减
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)

        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])

        return k
    

if __name__ == "__main__":
    np.random.seed(1)  # 设定随机种子,使实验具有可重复性
    K = 10
    bandit_10_arm = BernoulliBandit(K)
    print("随机生成了一个%d臂伯努利老虎机" % K)
    print("获奖概率最大的拉杆为%d号,其获奖概率为%.4f" %
        (bandit_10_arm.best_idx, bandit_10_arm.best_prob))
    np.random.seed(1)
    # epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
    # epsilon_greedy_solver.run(5000)
    # print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)
    # plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])
    decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
    decaying_epsilon_greedy_solver.run(5000)
    print('epsilon值衰减的贪婪算法的累积懊悔为：', decaying_epsilon_greedy_solver.regret)
    plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])
    print("end")