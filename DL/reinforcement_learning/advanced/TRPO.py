#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   TRPO.py
@Time   :   2024/08/26 11:50:10
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   TRPO代码
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')


import torch
import numpy as np
import gym
import matplotlib.pyplot as plt
import torch.nn.functional as F
import rl_utils
import copy

from Actor_Critic import PolicyNet, ValueNet


def compute_advantage(gamma, lmbda, td_delta: torch.tensor):
    """GAE

    Args:
        td_delta: 每个时间步的delta(时序差分误差)
    """
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


class TRPO:
    def __init__(
        self,
        hidden_dim,
        state_space,
        action_space,
        lmbda,
        kl_constraint,
        alpha,
        critic_lr,
        gamma,
        device
    ):
        state_dim = state_space.shape[0]
        try:
            action_dim = action_space.n
        except:
            action_dim = action_space.shape[0]
        # 策略网络参数不需要优化器更新
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda  # GAE参数
        self.kl_constraint = kl_constraint  # KL距离最大限制
        self.alpha = alpha  # 线性搜索参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
    def hessian_matrix_vector_product(self, states, old_action_dists, vector):
        """计算KL散度海森矩阵和一个向量的乘积
        即先计算Hx再对其求一阶导数
        """
        new_action_dists = torch.distributions.Categorical(self.actor(states))
        kl = torch.mean(
            torch.distributions.kl.kl_divergence(
                old_action_dists,
                new_action_dists
            )
        )
        kl_grad = torch.autograd.grad(
            kl,
            self.actor.parameters(),
            create_graph=True
        )
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        # KL距离的梯度先和向量进行点积运算
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(
            kl_grad_vector_product,
            self.actor.parameters()
        )
        grad2_vector = torch.cat([grad.view(-1) for grad in grad2])
        return grad2_vector

    def conjugate_gradient(self, grad, states, old_action_dists):
        """共轭梯度求解
        """
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        rdotr = torch.dot(r, r)
        # 共轭梯度的主循环
        for i in range(10):
            Hp = self.hessian_matrix_vector_product(
                states,
                old_action_dists,
                p
            )
            alpha = rdotr / torch.dot(p, Hp)
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x
    
    def compute_surrogate_obj(
        self, states, actions, advantage,
        old_log_probs, actor
    ):
        """计算策略目标L(θ)
        """
        log_probs = torch.log(actor(states).gather(1, actions))
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)

    def line_search(
        self, states, actions, advantage,
        old_log_probs, old_action_dists, max_vec
    ):
        """线性搜索

        Args:
            max_vec: 笔记里面根号里面那一大堆
        """
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(
            self.actor.parameters()
        )
        old_obj = self.compute_surrogate_obj(
            states, actions, advantage,
            old_log_probs, self.actor
        )
        for i in range(15):
            coef = self.alpha**i
            new_para = old_para + coef * max_vec
            new_actor = copy.deepcopy(self.actor)
            # IMPORTANT 快速替换参数
            torch.nn.utils.convert_parameters.vector_to_parameters(
                new_para, new_actor.parameters()
            )
            new_action_dists = torch.distributions.Categorical(
                new_actor(states)
            )
            kl_div = torch.mean(
                torch.distributions.kl.kl_divergence(
                    old_action_dists,
                    new_action_dists
                )
            )
            new_obj = self.compute_surrogate_obj(
                states, actions, advantage,
                old_log_probs, new_actor
            )
            if new_obj > old_obj and kl_div < self.kl_constraint:
                return new_para
        return old_para

    def policy_learn(
        self, states, actions, advantage,
        old_action_dists, old_log_probs,
    ):
        """更新策略函数
        """
        surrogate_obj = self.compute_surrogate_obj(
            states, actions, advantage,
            old_log_probs, self.actor
        )
        grads = torch.autograd.grad(
            surrogate_obj,
            self.actor.parameters()
        )
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        # 用共轭梯度法计算x = H^(-1)g
        descent_direction = self.conjugate_gradient(
            obj_grad, states,
            old_action_dists
        )
        # x就是descent_direction
        Hd = self.hessian_matrix_vector_product(
            states,
            old_action_dists,
            descent_direction
        )
        # FIXME max_coef为nan
        max_coef = torch.sqrt(
            2 * self.kl_constraint /  (torch.dot(descent_direction, Hd) + 1e-8)
        )
        # 线性搜索
        new_para = self.line_search(
            states, actions, advantage,
            old_log_probs, old_action_dists,
            descent_direction * max_coef
        )
        # 新参数替换
        torch.nn.utils.convert_parameters.vector_to_parameters(
            new_para, self.actor.parameters()
        )

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
        old_action_dists = torch.distributions.Categorical(self.actor(states).detach())
        critic_loss = torch.mean(
            F.mse_loss(
                self.critic(states), td_target.detach()
            )
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()  # 更新价值函数
        # 更新策略函数
        self.policy_learn(
            states, actions, advantage,
            old_action_dists, old_log_probs,
        )


class PolicyNetContinuous(torch.nn.Module):
    """因为与之交互的是倒立摆环境，这个环境需要连续的动作空间
    """
    def __init__(self, state_dim, hidden_dim, action_dim) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """输出动作的高斯均值和标准差
        """
        logits = self.fc1(x)
        x = F.relu(logits)
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std


class TRPOContinuous(TRPO):
    """处理连续动作的TRPO算法
    """
    def __init__(
        self,
        hidden_dim,
        state_space,
        action_space,
        lmbda,
        kl_constraint,
        alpha,
        critic_lr,
        gamma,
        device
    ):
        super().__init__(
            hidden_dim, state_space, action_space,
            lmbda, kl_constraint, alpha, critic_lr,
            gamma, device
        )
        self.actor = PolicyNetContinuous(
            state_dim=state_space.shape[0],
            hidden_dim=hidden_dim,
            action_dim=action_space.shape[0]
        ).to(device)

    def take_action(self, state):
        """按照高斯均值标准差采样
        """
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        mu, std = self.actor(state)
        action_dist = torch.distributions.Normal(mu, std)
        action = action_dist.sample()
        return [action.item()]

    def hessian_matrix_vector_product(
        self,
        states,
        old_action_dists,
        vector,
        damping=0.1
    ):
        new_actions_dists = torch.distributions.Normal(
            *self.actor(states)
        )
        kl= torch.mean(
            torch.distributions.kl.kl_divergence(
                old_action_dists,
                new_actions_dists
            )
        )
        kl_grad = torch.autograd.grad(
            kl,
            self.actor.parameters(),
            create_graph=True
        )
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(
            kl_grad_vector_product,
            self.actor.parameters()
        )
        grad2_vector = torch.cat(
            [grad.contiguous().view(-1) for grad in grad2]
        )
        # FIXME ????
        return grad2_vector * damping * vector
    
    def compute_surrogate_obj(
        self, states, actions, advantage,
        old_log_probs, actor
    ):
        mu, std = actor(states)
        action_dists = torch.distributions.Normal(mu, std)
        log_probs = action_dists.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)

    def line_search(
        self, states, actions, advantage,
        old_log_probs, old_action_dists, max_vec
    ):
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(
            self.actor.parameters()
        )
        old_obj = self.compute_surrogate_obj(
            states, actions, advantage,
            old_log_probs, self.actor
        )
        for i in range(10):
            coef = self.alpha**i
            new_para = old_para + coef * max_vec
            new_actor = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(
                new_para,
                new_actor.parameters()
            )
            mu, std = new_actor(states)
            new_action_dists = torch.distributions.Normal(mu, std)
            kl_div = torch.mean(
                torch.distributions.kl.kl_divergence(
                    old_action_dists,
                    new_action_dists
                )
            )
            new_obj = self.compute_surrogate_obj(
                states, actions, advantage,
                old_log_probs, new_actor
            )
            if new_obj > old_obj and kl_div < self.kl_constraint:
                return new_para
        return old_para
    
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards = (rewards + 8.0) / 8.0  # 方便训练
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advanatage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        mu, std = self.actor(states)
        # IMPORTANT 原来离散直接通过gather获取条件概率，这边是log_prob
        # IMPORTANT 原来dists和prob是可以分别算，这边是前后关系
        old_action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_probs = old_action_dists.log_prob(actions)
        critic_loss = torch.mean(
            F.mse_loss(
                self.critic(states),
                td_target.detach()
            )
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.policy_learn(
            states, actions, advanatage,
            old_action_dists, old_log_probs
        )



if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # =============================== 离散环境 ==================================
    # num_episodes = 500
    # hidden_dim = 128
    # gamma = 0.98
    # lmbda = 0.95
    # critic_lr = 1e-2
    # kl_constraint = 0.0005
    # alpha = 0.5

    # env_name = 'CartPole-v1'
    # env = gym.make(env_name)
    # torch.manual_seed(0)
    # agent = TRPO(
    #     hidden_dim, env.observation_space, env.action_space, lmbda,
    #     kl_constraint, alpha, critic_lr, gamma, device
    # )
    # return_list = rl_utils.train_on_policy_agent(env, 0, agent, num_episodes)

    # episodes_list = list(range(len(return_list)))
    # plt.plot(episodes_list, return_list)
    # plt.xlabel('Episodes')
    # plt.ylabel('Returns')
    # plt.title('TRPO on {}'.format(env_name))
    # plt.show()

    # mv_return = rl_utils.moving_average(return_list, 9)
    # plt.plot(episodes_list, mv_return)
    # plt.xlabel('Episodes')
    # plt.ylabel('Returns')
    # plt.title('TRPO on {}'.format(env_name))
    # plt.show()

    # =============================== 连续环境 ==================================
    num_episodes = 2000
    hidden_dim = 128
    gamma = 0.9
    lmbda = 0.9
    critic_lr = 1e-2
    kl_constraint = 0.00005
    alpha = 0.5

    env_name = 'Pendulum-v1'
    env = gym.make(env_name, max_episode_steps=200)
    torch.manual_seed(0)
    agent = TRPOContinuous(
        hidden_dim, env.observation_space, env.action_space,
        lmbda, kl_constraint, alpha, critic_lr, gamma, device
    )
    return_list = rl_utils.train_on_policy_agent(env, 0, agent, num_episodes)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('TRPO on {}'.format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('TRPO on {}'.format(env_name))
    plt.show()
    print()