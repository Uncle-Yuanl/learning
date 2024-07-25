#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   Qlearning.py
@Time   :   2024/07/24 15:23:47
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   基础Q-learning
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from sarsa import CliffWalkingEnv, Sarsa, print_agent

class QLearning(Sarsa):
    """Q-learning算法
    完全继承了Sarsa的take_action方法，行为策略二者是一致的
    """
    def update(self, s0, a0, r, s1):
        """目标策略更新
        """
        td_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error


if __name__ == "__main__":
    ncol = 12
    nrow = 4
    env = CliffWalkingEnv(ncol, nrow)
    np.random.seed(0)
    alpha = 0.1
    epsilon = 0.1
    gamma = 0.9
    agent = QLearning(ncol, nrow, epsilon, alpha, gamma)
    num_episodes = 500  # 智能体在环境中运行的序列的数量

    # 记录每一条序列的回报
    # 注意：这是行为策略的回报，而不是目标策略
    return_list = []
    for i in range(10):  # 显示10个进度条
        # tqdm的进度条功能
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
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

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Q-learning on {}'.format('Cliff Walking'))
    plt.show()

    action_meaning = ['^', 'v', '<', '>']
    print('Q-learning算法最终收敛得到的策略为：')
    print_agent(agent, env, action_meaning, list(range(37, 47)), [47])