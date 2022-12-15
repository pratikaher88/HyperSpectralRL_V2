from agents.ac_agent import ACAgent 

import scipy
import numpy as np
import pandas as pd
import torch.nn as nn
from scipy.stats.stats import pearsonr
import torch
from matplotlib.pyplot import figure
from utils import from_numpy, to_numpy, DataManager, ReplayBuffer, LogManager

class RL_Trainer():

    def __init__(self, params, external_cache = {}):

        self.LogManager = LogManager(params)
        
        self.agent_params = params['agent']
        self.n_iter = self.agent_params['n_iter']

        self.trajectory_sample_size = self.agent_params['trajectory_sample_size']
        self.num_bands = self.agent_params['num_bands']

        self.data_params = params['data']
        self.band_selection_num = self.data_params['band_selection_num']

        if self.agent_params['reward_type'] == 'correlation':
            self.reward_func = self.calculate_correlations
        elif self.agent_params['reward_type'] == 'mutual_info':
            self.reward_func = self.calculate_mutual_infos
        
        self.data_params = params['data']
        
        self.DataManager = DataManager(self.data_params, self.num_bands)

        self.replay_buffer = ReplayBuffer()
        self.LogManager.logging_df = pd.DataFrame()
        self.cache = external_cache

        # agent_class = self.agent_params['agent_class']
        # self.agent = agent_class(self, params)

        if self.agent_params['agent_class'] == 'AC':
            agent_class = ACAgent
        
        self.agent = agent_class(self, params)

    def run_training_loop(self):
        
        prev_selected_bands = np.zeros(self.num_bands)

        for iter_num in range(self.n_iter):
            print('Iteration ', iter_num, ':')

            paths = self.generateTrajectories()
            self.replay_buffer.add_trajectories(paths)

            critic_loss = self.agent.train()

            print('------------------------------------EVAL Results------------------------------')
            eval_path = self.sampleTrajectory(iter_num)
            current_selected_bands = np.argwhere(eval_path[-1]['ob_next']>0).flatten()
             
            print('Selected_Bands: ', current_selected_bands)
            print('Common to previous array', np.intersect1d(current_selected_bands, prev_selected_bands).shape[0])
            
            print('Num_Selected_Bands: ', np.argwhere(eval_path[-1]['ob_next']>0).shape[0])
            print('Eval_Return: ', np.sum(eval_path[-1]['re']))
            print('Critic_Loss: ', critic_loss)
            # print('Correlation: ', self.LogManager.logging_df.loc[self.LogManager.logging_df.shape[0]-1, 'Correlation Next State'])
            
            prev_selected_bands = current_selected_bands


    def generateTrajectories(self):
        
        #we expect paths to be a list of trajectories
        #a trajectory is a list of Path objects
        paths = []
        for i in range(self.trajectory_sample_size):
            path = self.sampleTrajectory()
            paths.append(path)
    
        return paths
    
    def sampleTrajectory(self, iter_num = 1):
            
        #select 30 actions
        state = np.zeros(self.num_bands)
        state_next = state.copy()
        
        #paths will be a list of dictionaries
        path = []
        for i in range(self.band_selection_num):
            
            action = self.agent.policy.get_action(state)
            state_next[action] += 1

            reward, correlation_current_state, correlation_next_state = self.calculate_reward(state, state_next)

            terminal = 1 if i == self.band_selection_num - 1 else 0
            path.append(self.Path(state.copy(), action, state_next.copy(), reward, terminal))
            
            state = state_next.copy()
        
            # if iter_num % 25 == 0:
            #     q_values = self.agent.critic_policy.forward(state)
                
            #     sampled_paths = self.replay_buffer.sample_buffer_random(1)
                
            #     flat_sampled_path = [path for trajectory in sampled_paths for path in trajectory]
            #     obs = np.array([path['ob'] for path in flat_sampled_path])
            #     acs = np.array([path['ac'] for path in flat_sampled_path])
            #     obs_next = np.array([path['ob_next'] for path in flat_sampled_path])
            #     res = np.array([path['re'] for path in flat_sampled_path])
            #     terminals = np.array([path['terminal'] for path in flat_sampled_path])
                
            #     loss_value = self.agent.critic_policy.update(obs, acs, obs_next, res, terminals)
                
            #     row = {
            #         "iter_num": iter_num,
            #         "Selected Band": i,
            #         # "Action Type": action_type,
            #         "Mean": torch.mean(q_values).detach().numpy(),
            #         "Min": torch.min(q_values).detach().numpy(),
            #         "Max": torch.max(q_values).detach().numpy(),
            #         "Correlation Current State" : correlation_current_state,
            #         "Correlation Next State" : correlation_next_state,
            #         "Reward" : reward,
            #         "Loss" : loss_value
            #     }
                
            #     self.LogManager.logging_df = self.LogManager.logging_df.append(row, ignore_index=True)
                     
        return path

    def calculate_reward(self, state, state_next):
        #for future, save down the previous state so that we can avoid a calc
        
#         print(list(np.argwhere(np.array(state) != 0)), list(np.argwhere(np.array(state_next) != 0)))
        if list(np.argwhere(np.array(state) != 0)) == list(np.argwhere(np.array(state_next) != 0)):
#             print("same action selected")
            return -1, "Indef", "Indef"
        else:
        
            a = self.reward_func(state)
            b = self.reward_func(state_next)
            #FLIPPED THE SIGN FOR TESTING
            return a-b, a, b
#             return np.exp(a-b), a, b
    
    
    def calculate_correlations(self, state):
        
#         if repr(state) in self.cache:
#             return self.cache[repr(state)]
        
        #deal with the first state
        ##### THIS LOGIC SEEMS WRONG - REGARDLESS OF THE FIRST PICK, YOU HAVE A REWARD OF 0#####
        if np.sum(state) <= 1:
            return 0
        
        selected_bands = []
        non_zero_bands = np.argwhere(np.array(state) != 0)
        for band in non_zero_bands:
#             print(band[0])
            selected_bands.extend([band[0]]*int(state[band[0]]))
        #print(selected_bands)
        #selected_bands = np.squeeze(np.argwhere(np.array(state)==1))
        corr_sum = 0
        for idx_i, i in enumerate(selected_bands):
            for idx_j, j in enumerate(selected_bands):
                if idx_i != idx_j:
                    
                    if repr((i,j)) in self.cache:
                        result = self.cache[repr((i,j))]
                    else:
                        result = abs(pearsonr(self.DataManager.rl_data[:, i], self.DataManager.rl_data[:, j])[0])
                        self.cache[repr((i,j))] = result
                    
                    corr_sum += result
                    
#                     corr_sum += abs(pearsonr(self.DataManager.rl_data[:, i], self.DataManager.rl_data[:, j])[0])
        
#         self.cache[repr(state)] = corr_sum/(len(selected_bands)**2)
        
#         return self.cache[repr(state)]
        return corr_sum/(len(selected_bands)**2)

    
    def Path(self, ob, ac, ob_next, re, terminal):
        return {'ob':ob,
                'ac':ac,
                'ob_next':ob_next,
                're':re,
                'terminal':terminal
                }