'''
Class definitions for the Actor and Critic for the ddpg network for the Reacher problem
in the Udacity Deep Learning Nano degree course.

Some methods taken from DDPG file in Udacity Deep Learning course and previous p2 project
Coded by: Jeremy Vila
12/15/18
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Function taken from previous ddpg implementation in p2
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

# Actor based on previous implementation in p2
# class definition for the actor
class Actor(nn.Module):

    def __init__(self, state_size, action_size, seed=0, fc_units=[50, 100]):

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
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        # Adding bias seems to help here ...
        #self.fc1.bias.data.fill_(0.05)
        #self.fc2.bias.data.fill_(0.05)
        # self.fc3.bias.data.fill_(-0.05)   

    #  Three layer network
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

# Critic implementation based on implementation in p2
# class definition for the critic
class Critic(nn.Module):

    def __init__(self, state_size, action_size, seed=0, fc_units=[50, 150, 50]):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fc_units[0])
        self.fc1 = nn.Linear(action_size + fc_units[0], fc_units[1])
        self.fc2 = nn.Linear(fc_units[1], fc_units[2])
        self.fc3 = nn.Linear(fc_units[2], 1)
        self.reset_param()

    # Reset paramers
    def reset_param(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        # Adding bias seems to help ...
        # self.fc1.bias.data.fill_(0.05)
        # self.fc2.bias.data.fill_(0.05)
        # self.fc3.bias.data.fill_(0.05) 

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Concatenate hidden layer and actions
        state, action = state.squeeze(), action.squeeze()
        x = F.relu(self.fcs1(state))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
