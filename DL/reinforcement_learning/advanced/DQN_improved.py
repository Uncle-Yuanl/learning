#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   DQN_improved.py
@Time   :   2024/07/31 15:05:46
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   DQN的升级算法 DouleDQN DuelingDQN
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
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
from tqdm import tqdm

from DQN import DQN


class DoubleDQN(DQN):
    def max_q_value(self, state):
        """
        """
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        return self.q_net(state).max().item()
    
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        # 下个状态的最大Q值，使用训练网络根据s'计算a'
        max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
        # 区别就在这边：不是简单地.max(1)[0]，而是根据max_action选择
        max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        # TD误差目标
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict()
            )
        self.count += 1


def dis_to_con(discrete_action, env, action_dim):  # 离散动作转回连续的函数
    action_lowbound = env.action_space.low[0]  # 连续动作的最小值
    action_upbound = env.action_space.high[0]  # 连续动作的最大值
    return action_lowbound + (discrete_action /
                              (action_dim - 1)) * (action_upbound -
                                                   action_lowbound)


def train_DQN(agent, env, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    max_q_value_list = []
    max_q_value = 0
    for i in range(10):
        with tqdm(total=int(num_episodes / 10),
                  desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state, info = env.reset(seed=0)
                done = False
                truncated = False
                while not done and not truncated:
                    action = agent.take_action(state)
                    max_q_value = agent.max_q_value(
                        state) * 0.005 + max_q_value * 0.995  # 平滑处理
                    max_q_value_list.append(max_q_value)  # 保存每个状态的最大Q值
                    # action_continuous = dis_to_con(action, env, agent.action_dim)
                    # next_state, reward, done, truncated, info = env.step([action_continuous])
                    next_state, reward, done, truncated, info = env.step([action])
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(
                            batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return return_list, max_q_value_list


# =========================================================================
# =========================== Dueling DQN =================================
# =========================================================================
class VAnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_V = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        share = F.relu(self.fc1(x))
        A = self.fc_A(share)
        V = self.fc_V(share)
        Q = V + A + A.mean(1).view(-1, 1)
        return Q


if __name__ == "__main__":
    lr = 1e-2
    num_episodes = 200
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 50
    buffer_size = 5000
    minimal_size = 1000
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env_name = 'Pendulum-v1'
    env = gym.make(env_name, max_episode_steps=200)
    state_dim = env.observation_space.shape[0]
    action_dim = 11  # 将连续动作分成11个离散动作

    random.seed(0)
    np.random.seed(0)
    # env.seed(0)
    torch.manual_seed(0)
    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    agent = DoubleDQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                target_update, device)
    return_list, max_q_value_list = train_DQN(agent, env, num_episodes,
                                            replay_buffer, minimal_size,
                                            batch_size)

    episodes_list = list(range(len(return_list)))
    mv_return = rl_utils.moving_average(return_list, 5)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.show()

    frames_list = list(range(len(max_q_value_list)))
    plt.plot(frames_list, max_q_value_list)
    plt.axhline(0, c='orange', ls='--')
    plt.axhline(10, c='red', ls='--')
    plt.xlabel('Frames')
    plt.ylabel('Q value')
    plt.title('DQN on {}'.format(env_name))
    plt.show()