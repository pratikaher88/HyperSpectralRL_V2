from torch import distributions
from torch import optim
import numpy as np
import torch
import torch.nn as nn
from utils import build_mlp, from_numpy, to_numpy, DataManager, ReplayBuffer
from critics.qcritic import QCritic
import torch.nn.functional as F

class CriticPolicy():
    def __init__(self, critic_params):
        
        self.critic_params = critic_params
        self.num_bands = self.critic_params['num_bands']
        self.learning_rate = self.critic_params['learning_rate']
        self.num_grad_steps_per_target_update = self.critic_params['num_grad_steps_per_target_update']
        self.num_target_updates = self.critic_params['num_target_updates']

        # self.critic_network = self.create_network()
        self.critic_network = build_mlp(input_size=self.num_bands,
                                    output_size=1,
                                    n_layers=2,
                                    size=64,
                                    activation='linear')
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.critic_network.parameters(),
            self.learning_rate
        )
        
        self.gamma = critic_params['gamma']

    
    def get_action(self, obs):
        if isinstance(obs, np.ndarray):
            obs = from_numpy(obs)

        return self.critic_network(obs)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = from_numpy(obs)

        return self.critic_network(obs).squeeze(1)
    
    def forward_np(self, obs):
        obs = from_numpy(obs)
        predictions = self.forward(obs)
        return to_numpy(predictions)
    
    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        
        ob_no = from_numpy(ob_no)
        ac_na = from_numpy(ac_na)
        next_ob_no = from_numpy(next_ob_no)
        reward_n = from_numpy(reward_n)
        terminal_n = from_numpy(terminal_n)

        for i in range(self.num_grad_steps_per_target_update):
            next_v = self.forward(next_ob_no)
            target = reward_n + self.gamma * next_v * (1 - terminal_n)
            # TODO : assert that check sizes

            for j in range(self.num_target_updates):
                pred = self.forward(ob_no)

                assert pred.shape == target.shape
                loss = self.loss(pred, target.detach())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return loss.item()

    
    def create_network(self):
        
        q_net  = nn.Sequential(
        nn.Linear(self.num_bands, self.num_bands*2),
        nn.ReLU(),
        nn.Linear(self.num_bands*2, self.num_bands*2),
        nn.ReLU(),
        nn.Linear(self.num_bands*2, 1)
        )
        
        return q_net
        

class ActorPolicy():
    
    def __init__(self, actor_params):
        self.actor_params = actor_params

        self.num_bands = self.actor_params['num_bands']
        self.band_selection_num = self.actor_params['band_selection_num']
        self.learning_rate = self.actor_params['learning_rate']
        self.epsilon = actor_params['epsilon']
        self.epsilon_decay = actor_params['epsilon_decay']

        # self.logits_na = self.create_network()
        self.logits_na = build_mlp(input_size=self.num_bands,
                                    output_size=self.num_bands,
                                    n_layers=2,
                                    size=64,
                                    activation='softmax')

        self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        
    def create_network(self):
        
        q_net  = nn.Sequential(
        nn.Linear(self.num_bands, self.num_bands*2),
        nn.ReLU(),
        nn.Linear(self.num_bands*2, self.num_bands*2),
        nn.ReLU(),
        nn.Linear(self.num_bands*2, self.num_bands)
        )
        
        return q_net
    
    def get_action(self, obs):
        
        if isinstance(obs, np.ndarray):
            obs = from_numpy(obs)
            
        # action_distribution = self.forward(obs)
        # action = action_distribution.sample()
        
        # return to_numpy(action), "Random"

        selected_bands = np.squeeze(np.argwhere(obs != 0)).tolist()
        number_of_non_zeros = np.count_nonzero(obs)
        
        rand = np.random.rand()
        if rand < self.epsilon or number_of_non_zeros <= 1:
            unselected_bands = np.squeeze(np.argwhere(obs == 0))
            selected_idx = np.random.choice(unselected_bands)
            action_type = "Random Action"
        else:
            action_distribution = self.forward(obs)
            selected_idx = action_distribution.sample()
            
            # print(selected_idx, selected_bands)
            count = 0
            unselected_bands = np.squeeze(np.argwhere(obs == 0))
            while selected_idx in selected_bands:
                selected_idx = action_distribution.sample()
                # TODO : FIx this if it slows down runs
                # if count > 10:
                #     selected_idx = np.random.choice(unselected_bands)
                # count += 1


            # print(selected_idx)

            action_type = "Sampled Action"
        
        self.decay_epsilon()
        return selected_idx, action_type
    
    def forward(self, observation):
        
        logits = self.logits_na(observation)
        action_distribution = distributions.Categorical(logits=logits)
        return action_distribution
    
    def update(self, observations, actions, adv_n=None, acs_labels_na=None, qvals=None):
        
        observations = from_numpy(observations)
        actions = from_numpy(actions)
        adv_n = from_numpy(adv_n)

        action_distribution = self.forward(observations)
        loss = - action_distribution.log_prob(actions) * adv_n
        loss = loss.sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay 
