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

class ArgMaxPolicy():
    
    def __init__(self, params, critic):
        self.epsilon = params['epsilon']
        self.epsilon_decay = params['epsilon_decay']
        self.critic = critic
        
    def get_action(self, obs):
        
        q_value_estimates = self.critic.get_action(obs)
        unselected_bands = np.squeeze(np.argwhere(obs == 0))
        #print(obs)
#         print('Predicted Q-Values:', q_value_estimates)
        
        number_of_non_zeros = np.count_nonzero(obs)
        
        rand = np.random.rand()
        if rand < self.epsilon or number_of_non_zeros <= 1:
            #select a random action
#             print('Selected Random')
            unselected_bands = np.squeeze(np.argwhere(obs == 0))
            selected_idx = np.random.choice(unselected_bands)
            action_type = "Random Action"

        else:
#           print('Selected Max')
            #q_value_estimates_idx = torch.argsort(q_value_estimates, dim=1)
            #q_value_estimates = q_value_estimates[unselected_bands, :]
        
            #q_filter = q_value_estimates[unselected_bands]
            q_value_estimates_idx = torch.argsort(q_value_estimates, descending=True)
            q_value_estimates_idx = q_value_estimates_idx[torch.isin(q_value_estimates_idx, torch.tensor(unselected_bands))]
            selected_idx = q_value_estimates_idx[0].item()
                
            action_type = "Max Action"

            
        self.decay_epsilon()
        return selected_idx, action_type
                
    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay 