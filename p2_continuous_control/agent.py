# This is the agent for the continuous control problem.
# The environment is defined by the Unity engine
#
# This agent controls a robotic arm with 12 degrees of freedom
#
# Coded by Jeremy Vila
# 10/20/18

import torch
import torch.nn as nn
import torch.nn.functional as functional

class Agent(nn.Module):
    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Agent, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        hidden_sizes = [32, 16, 8, 4]
        dropout_rates = [.2, .2]

        self.fc1 = nn.Linear(state_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_size[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc5 = nn.Linear(hidden_sizes[3], action_size)

        # Dropout considered, but not needed!
        self.drop1 = nn.Dropout(p=dropout_rates[0])
        self.drop2 = nn.Dropout(p=dropout_rates[1])

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        # x = self.drop1(x)
        x = F.relu(self.fc2(x))
        # x = self.drop2(x)
        return self.fc3(x)