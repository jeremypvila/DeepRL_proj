'''
Class definitions for the Actor and Critic for the ddpg network for the Reacher problem
in the Udacity Deep Learning Nano degree course.

Some methods taken from DDPG file in Udacity Deep Learning course
Coded by: Jeremy Vila
11/13/18
'''

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

# class definition for the actor
class Actor(nn.Module):

    def __init__(self, state_size, action_size, seed, fc_units=[256, 128]):

        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_units[0])
        self.fc2 = nn.Linear(fc_units[0], fc_units[1])
        self.fc3 = nn.Linear(fc_units[1], action_size)
        self.reset_param()  # Reset parameters for the model

    # method to have a hard coding of parameters
    def reset_param(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)
        # Adding bias seems to help here ...
        self.fc1.bias.data.fill_(0.05)
        self.fc2.bias.data.fill_(0.05)
        self.fc3.bias.data.fill_(0.05)   

    #  Three layer network
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

# class definition for the critic
class Critic(nn.Module):

    def __init__(self, state_size, action_size, seed, fc_units=[256, 256, 128]):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_units[0])
        self.fc2 = nn.Linear(fc_units[0] + action_size, fc_units[1])
        self.fc3 = nn.Linear(fc_units[1], fc_units[2])
        self.fc4 = nn.Linear(fc_units[2], 1)
        self.reset_param()  # Reset the parameters

    # Reset paramers
    def reset_param(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)
        # Adding bias seems to help ...
        self.fc1.bias.data.fill_(0.05)
        self.fc2.bias.data.fill_(0.05)
        self.fc3.bias.data.fill_(0.05) 
        self.fc4.bias.data.fill_(0.05)

    def forward(self, state, action):
        xs = F.relu(self.fc1(state))
        # Concatenate hidden layer and actions
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
