from torch import distributions
import scipy
import numpy as np
from collections import OrderedDict

import torch.nn as nn

import torch
from utils import from_numpy, to_numpy, DataManager, ReplayBuffer
from policies.ac_policy import ActorPolicy, CriticPolicy


class ACAgent():

    def __init__(self, parent, agent_params):
            
        critic_params = agent_params['critic']
        actor_params = agent_params['actor']
        self.agent_params = agent_params['agent']
        self.num_critic_updates = self.agent_params['num_critic_updates']
        self.gamma = critic_params['gamma']
        
        self.policy = ActorPolicy(actor_params)
        self.critic = CriticPolicy(critic_params)

        self.replay_buffer = parent.replay_buffer
        
    def train(self):

        for _ in range(self.num_critic_updates):

            sampled_paths = self.replay_buffer.sample_buffer_random(self.agent_params['batch_size'])

            flat_sampled_path = [path for trajectory in sampled_paths for path in trajectory]
            obs = np.array([path['ob'] for path in flat_sampled_path])
            acs = np.array([path['ac'] for path in flat_sampled_path])
            obs_next = np.array([path['ob_next'] for path in flat_sampled_path])
            res = np.array([path['re'] for path in flat_sampled_path])
            terminals = np.array([path['terminal'] for path in flat_sampled_path])
        
            for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
                critic_loss = self.critic.update(obs, acs, obs_next, res, terminals)

            advantages = self.estimate_advantage(obs, obs_next, res, terminals)

            for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
                actor_loss = self.policy.update(obs, acs, adv_n=advantages)

            loss = OrderedDict()
            loss['Critic_Loss'] = critic_loss
            loss['Actor_Loss'] = actor_loss

        return loss
    
    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        
        v_n = self.critic.forward_np(ob_no)
        next_v_n = self.critic.forward_np(next_ob_no)

        assert v_n.shape == next_v_n.shape == re_n.shape == terminal_n.shape

        q_n = re_n + self.gamma * next_v_n * (1 - terminal_n)
        adv_n = q_n - v_n
            
        return adv_n

