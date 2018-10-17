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

	def __init__(self):
		super(Agent, self).__init__()