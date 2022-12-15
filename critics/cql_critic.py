import torch.nn as nn
import torch
import numpy as np
from utils import from_numpy, to_numpy, DataManager, ReplayBuffer

class CQLCritic:

    def __init__(self, params):
        
        self.exploitation_params = params['exploitation']
        self.num_bands = params['num_bands']
        self.double_q = params['double_q']
        self.q_net = self.create_network()
        self.q_net_target = self.create_network()

        self.gamma = self.exploitation_params['gamma']
        self.optimizer = torch.optim.Adam(self.q_net_target.parameters(),lr=0.005)
        self.loss = nn.MSELoss()

        self.alpha = self.exploitation_params['alpha']
        



    def create_network(self):
        
        q_net  = nn.Sequential(
        nn.Linear(self.num_bands, self.num_bands*2),
        nn.ReLU(),
        nn.Linear(self.num_bands*2, self.num_bands*2),
        nn.ReLU(),
        nn.Linear(self.num_bands*2, self.num_bands)
        )
        
        return q_net


    def dqn_loss(self, obs, next_obs, reward_n, terminals):

        full_q_values = self.q_net(obs)
        q_actions = full_q_values.argmax(dim=1)
        q_values = torch.gather(full_q_values, 1, q_actions.unsqueeze(1)).squeeze(1)
        
        #print('Obs ', obs.shape)
        #print('Full Q ', full_q_values.shape)
        #print('Q Actions ', q_actions.shape)
        #print('Q Val ', q_values.shape)
        
        full_q_next_target = self.q_net_target(next_obs)

        if self.double_q:
            q_actions_next = self.q_net(next_obs).argmax(dim=1)
            q_values_next = torch.gather(full_q_next_target, 1, q_actions_next.unsqueeze(1)).squeeze(1)
        else:
            q_values_next, _ = full_q_next_target.max(dim=1)

        #q_values_next = full_q_next.max(dim=1)
        #print('q_values_next', q_values_next)
        
        
        #print('reward', type(reward_n.shape))
        #print('q_values_next', type(q_values_next))
        #print('terminals', type(terminals))
        #print('gamma', type(self.gamma))
        target = reward_n + self.gamma*q_values_next*(1-terminals)
        target = target.detach()
        
        #print(f'Target Dim: {target.shape}')
        #print(f'Q_Values Dim: {q_values.shape}')
        loss = self.loss(q_values, target)

        return loss, full_q_values, q_values

    def update(self, obs, ac_n, next_obs, reward_n, terminals):

        obs = self.check_tensor(obs) #comes in as shape 
        ac_n = self.check_tensor(ac_n)
        next_obs = self.check_tensor(next_obs)
        reward_n = self.check_tensor(reward_n)
        terminals = self.check_tensor(terminals)

        loss, full_q_values, q_values = self.dqn_loss(obs, next_obs, reward_n, terminals)

        q_logsumexp = torch.log(torch.sum(torch.exp(full_q_values), dim=1))
        cql_loss = loss + self.alpha*((q_logsumexp - q_values).mean())

        self.optimizer.zero_grad()
        cql_loss.backward()
        ## DO WE NEED TO ADD NORM CLIPPING
        self.optimizer.step()

        ##DO WE NEED TO DECREASE THE LEARNING RATE?

    def update_target_network(self):
        for target_param, param in zip(
                self.q_net_target.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(param.data)


    def check_tensor(self, ar):
        
        if isinstance(ar, np.ndarray):
            ar = from_numpy(ar)
            
        return ar

    def get_action(self, obs):
        
        if isinstance(obs, np.ndarray):
            obs = from_numpy(obs)
            
        return self.q_net(obs)





