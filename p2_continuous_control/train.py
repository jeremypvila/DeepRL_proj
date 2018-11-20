'''
Main script to train the sspg algoirthm on the Reacher problem.

Some methods taken from DDPG file in Udacity Deep Learning course
Coded by:
Jeremy Vila
11/13/18

'''

from unityagents import UnityEnvironment
from ddpg_agent import Agent
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import torch

# Load the environment 
env = UnityEnvironment(file_name='Reacher_Linux_NoVis/Reacher.x86_64')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# get number of agenets.  Confirm there is one
num_agents = len(env_info.agents)
# Get action size
action_size = brain.vector_action_space_size

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]

agent = Agent(state_size=state_size, action_size=action_size, random_seed=0)
# agent.load_weights()

episodes = 2000  # Number of episodes
max_time = 1000  # Max number of time steps per episode
max_score = 32.  # average score to beat

# Score lists
scores_deque = deque(maxlen=100)
all_scores = []

# Main training loop
for ep in range(0, episodes):
	env_info = env.reset(train_mode=True)[brain_name]
	states = env_info.vector_observations

	# state = env.reset()
	agent.reset()  # Resets the noise in the agent
	scores = np.zeros(num_agents)
	# Step through time steps and learn the actor and critic
	for t in range(max_time):

		actions = agent.act(states)									# Get actions from policy (for each agent)
		env_info = env.step(actions)[brain_name]           			# Perform actions in environment
		next_states = env_info.vector_observations					# get next state (for each agent)
		rewards = env_info.rewards                         			# get reward (for each agent)
		dones = env_info.local_done									# get dones (for each agent)
		agent.step(states, actions, rewards, next_states, dones)	# Add experience to buffer
		states = next_states										# Reset states
		scores += env_info.rewards									# Accumulate rewards
		if np.any(dones):
			break 

	scores_deque.append(np.mean(scores))
	all_scores.append(np.mean(scores))  # Add to total list of scores

	# Print results as they are computed
	mn_score = np.mean(scores_deque)
	print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(ep+1, mn_score, np.mean(scores)), end="")
	if ep+1 % 100 == 0 or mn_score > max_score:
		torch.save({'local': agent.actor_local.state_dict(),
					'target': agent.actor_target.state_dict(),
					'opt': agent.actor_optimizer.state_dict()}, 'cc_actor.pth')
		torch.save({'local': agent.critic_local.state_dict(),
					'target': agent.critic_target.state_dict(),
					'opt': agent.critic_optimizer.state_dict()}, 'cc_critic.pth')
	if mn_score > max_score:
		break

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(all_scores)), all_scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig("trained_scores.png")
