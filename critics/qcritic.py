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

class QCritic():
    
    def __init__(self, params, num_bands):
        
        self.num_bands = num_bands
        
        self.critic = self.create_network()
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=0.005)

        self.critic_target = self.create_network()
        
        self.gamma = params['gamma']
        
        self.loss = nn.SmoothL1Loss()
    
    def create_network(self):
        
        q_net  = nn.Sequential(
        nn.Linear(self.num_bands, self.num_bands*2),
        nn.ReLU(),
        nn.Linear(self.num_bands*2, self.num_bands*2),
        nn.ReLU(),
        nn.Linear(self.num_bands*2, self.num_bands)
        )
        
        return q_net
    
        
    def forward(self, obs):
        # will take in one hot encoded states and output a list of qu values
        
        q_values = self.critic(obs)
        
        return q_values
    
    def get_action(self, obs):
        
        if isinstance(obs, np.ndarray):
            obs = from_numpy(obs)
            
        return self.critic(obs)
    
    def update(self, obs, ac_n, next_obs, reward_n, terminals):
        
        obs = self.check_tensor(obs) #comes in as shape 
        ac_n = self.check_tensor(ac_n)
        next_obs = self.check_tensor(next_obs)
        reward_n = self.check_tensor(reward_n)
        terminals = self.check_tensor(terminals)
        
        full_q_values = self.critic(obs)
        q_actions = full_q_values.argmax(dim=1)
        q_values = torch.gather(full_q_values, 1, q_actions.unsqueeze(1)).squeeze(1)
        
        
        #print('Obs ', obs.shape)
        #print('Full Q ', full_q_values.shape)
        #print('Q Actions ', q_actions.shape)
        #print('Q Val ', q_values.shape)
        
        full_q_next_target = self.critic_target(next_obs)
        q_actions_next = self.critic(next_obs).argmax(dim=1)
        #q_values_next = full_q_next.max(dim=1)
        #print('q_values_next', q_values_next)
        q_values_next = torch.gather(full_q_next_target, 1, q_actions_next.unsqueeze(1)).squeeze(1)
        
        #print('reward', type(reward_n.shape))
        #print('q_values_next', type(q_values_next))
        #print('terminals', type(terminals))
        #print('gamma', type(self.gamma))
        target = reward_n + self.gamma*q_values_next*(1-terminals)
        target = target.detach()
        
        #print(f'Target Dim: {target.shape}')
        #print(f'Q_Values Dim: {q_values.shape}')
        loss = self.loss(q_values, target)
        
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        
        return loss.item()
    
    def check_tensor(self, ar):
        
        if isinstance(ar, np.ndarray):
            ar = from_numpy(ar)
            
        return ar
    
    def update_target_network(self):
        for target_param, param in zip(
            self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

