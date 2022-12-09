from sklearn.linear_model import LogisticRegression
import scipy.io
import scipy
import numpy as np
import h5py
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import torch.nn as nn
from scipy.stats.stats import pearsonr
import torch
from matplotlib.pyplot import figure
import pickle
from utils import from_numpy, to_numpy, DataManager, ReplayBuffer
from critics.qcritic import QCritic
from policies.argmax_policy import ArgMaxPolicy

class DQNAgent():
    
    def __init__(self, parent, params):
        
        self.agent_params = params['agent']
        self.batch_size = self.agent_params['batch_size']
        self.num_critic_updates = self.agent_params['num_critic_updates']
        
        valid_rewards = ['correlation', 'mutual_info']
        assert self.agent_params['reward_type'] in valid_rewards, 'rewards must be one of ' + valid_rewards.join(',') 
        
        self.num_bands = self.agent_params['num_bands']

        self.critic_params = params['critic']
        self.critic = QCritic(self.critic_params, self.num_bands)
        
        
        self.policy_params = params['policy']
        self.policy = ArgMaxPolicy(self.policy_params, self.critic)
        
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
            
            critic_loss = self.critic.update(obs, acs, obs_next, res, terminals)
            
        self.critic.update_target_network()

        return critic_loss


    # def calculate_mutual_infos(self, state):
    
    #     selected_bands = []
    #     non_zero_bands = np.argwhere(np.array(state) != 0)
    #     for band in non_zero_bands:
    #         selected_bands.extend([band[0]]*int(state[band[0]]))
    
    #     normalized_mutual_info_score_sum = 0
    #     for i in selected_bands:
    #         for j in selected_bands:
                
    #             if i != j:

    #                 normalized_mutual_info_score_sum += normalized_mutual_info_score(self.DataManager.rl_data[:, i],
    #                                                                                  self.DataManager.rl_data[:, j])

    #     return normalized_mutual_info_score_sum/(len(selected_bands)**2)
