#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   Dyna_Q.py
@Time   :   2024/07/25 12:24:41
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   实现Dyna-Q算法
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')
import random
from tqdm import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt

from sarsa import CliffWalkingEnv
from Qlearning import QLearning


class DynaQ(QLearning):
    def __init__(self, n_planning, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.n_planning = n_planning  #执行Q-planning的次数, 对应1次Q-learning
        self.model = dict()  # 环境模型

    def q_learning(self, s0, a0, r, s1):
        """QLearning的update
        """
        super().update(s0, a0, r, s1)

    def update(self, s0, a0, r, s1):
        """Dyna-Q的更新逻辑
        """
        # 处理真实环境交互的结果
        self.q_learning(s0, a0, r, s1)
        # 将该样本加入模型（数据）
        self.model[(s0, a0)] = r, s1
        for _ in range(self.n_planning):
            # 随机取历史数据的状态动作对
            (s, a), (r, s_) = random.choice(list(self.model.items()))
            self.q_learning(s, a, r, s_)


def DynaQ_CliffWalking(n_planning):
    """Dyna-Q 算法在悬崖漫步环境中的训练函数
    为了方便对比不同n_planning参数对训练的影响
    """
    ncol = 12
    nrow = 4
    env = CliffWalkingEnv(ncol, nrow)
    epsilon = 0.01
    alpha = 0.1
    gamma = 0.9
    agent = DynaQ(n_planning, ncol, nrow, epsilon, alpha, gamma)
    num_episodes = 300  # 智能体在环境中运行多少条序列

    return_list = []  # 记录每一条序列的回报
    for i in range(10):  # 显示10个进度条
        # tqdm的进度条功能
        with tqdm(total=int(num_episodes / 10),
                  desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done = env.step(action)
                    episode_return += reward  # 这里回报的计算不进行折扣因子衰减
                    agent.update(state, action, reward, next_state)
                    state = next_state
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return return_list



if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    n_planning_list = [0, 2, 20]
    for n_planning in n_planning_list:
        print('Q-planning步数为：%d' % n_planning)
        time.sleep(0.5)
        return_list = DynaQ_CliffWalking(n_planning)
        episodes_list = list(range(len(return_list)))
        plt.plot(episodes_list,
                return_list,
                label=str(n_planning) + ' planning steps')
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Dyna-Q on {}'.format('Cliff Walking'))
    plt.show()