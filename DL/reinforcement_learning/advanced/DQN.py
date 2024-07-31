#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   DOQ.py
@Time   :   2024/07/29 11:17:53
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   实现DQN
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')


import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from rl_utils import ReplayBuffer, train_off_policy, moving_average


class Qnet(torch.nn.Module):
    """定义只有一层隐藏层的Q网络
    """
    def __init__(self, state_dim, hidden_dim, action_dim) -> None:
        """
        Args:
            state_dim: 输入维度
            hidden_dim： 隐层维度
            action_dim： 输出维度
        """
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    

class DQN:
    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        learning_rate,
        gamma,
        epsilon,
        target_update,
        device
    ):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        # Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率, 笔记中的C
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):
        """epsilon-贪婪策略采取动作
        """
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action
    
    def update(self, transition_dict):
        """
        Args:
            transition_dict: buffer的四元组转成的数据字典
        """
        # torch tensor化
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        # Q值 Q_w(s_i, a_i)
        q_values = self.q_net(states).gather(1, actions)
        # 下个状态的最大Q值 Q_w\bar(s', a')
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        # TD误差目标
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        # 均方误差损失函数
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        self.optimizer.zero_grad()
        # 梯度反向传播
        dqn_loss.backward()
        # 参数更新
        self.optimizer.step()
        # 更新更新步数，判断更新目标网络
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict()
            )
        self.count += 1


if __name__ == "__main__":
    lr = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env_name = "CartPole-v1"
    env = gym.make(env_name)
    random.seed(0)
    np.random.seed(0)
    # env.seed(0)  # 貌似启用了,现在在reset函数中作为参数
    envseed = 0
    torch.manual_seed(0)

    buffer = ReplayBuffer(buffer_size)
    replay_buffer = ReplayBuffer(buffer_size)
    # 描述环境的维度，每个维度都是连续的
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(
        state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
        target_update, device
    )

    return_list = train_off_policy(env, envseed, agent, num_episodes, replay_buffer, minimal_size, batch_size)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.show()

    mv_return = moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.show()

    print()