
from cmath import tanh
#import torch.nn as nn
import torch
from torch import nn
from utils import *


def init_method_1(model):
    model.weight.data.uniform_()
    model.bias.data.uniform_()

def init_method_2(model):
    model.weight.data.normal_()
    model.bias.data.normal_()

class RNDPolicy():

    def __init__(self, num_bands, **kwargs):
        super().__init__()
        self.num_bands = num_bands
        self.network_one = self.create_network(innit_method=init_method_1)
        self.network_two = self.create_network(innit_method=init_method_2)

        self.optimizer = torch.optim.Adam(self.network_two.parameters(),lr=0.005)

    def create_network(self, innit_method):
        net = nn.Sequential(
        nn.Linear(self.num_bands, self.num_bands*2).apply(innit_method),
        nn.Tanh(),
        nn.Linear(self.num_bands*2, self.num_bands*2).apply(innit_method),
        nn.Tanh(),
        nn.Linear(self.num_bands*2, self.num_bands).apply(innit_method),
        nn.Identity()
        )
        return net

    def forward(self, obs):

        obs = check_tensor(obs)

        network_one_pred = self.network_one(obs).detach()
        network_two_pred = self.network_two(obs)

        return torch.norm(network_two_pred - network_one_pred, dim=1)

    def update(self, obs):
        
        
        error = self.forward(obs)

        loss = torch.mean(error)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
        


