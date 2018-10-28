'''
Main script that was used for laoding the environment

'''

from unityagents import UnityEnvironment
from ddpg_agent import Agent
from collections import deque
import numpy as np 

# Load the environment 
env = UnityEnvironment(file_name='/home/jeremy/projects/DeepRL_proj/p2_continuous_control/Reacher_Linux_NoVis/Reacher.x86_64')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

print(state_size)
print(action_size)
agent = Agent(state_size=state_size, action_size=action_size, random_seed=0)

episodes = 100
max_time = 1000

scores_deque = deque(maxlen=100)
max_score = -np.Inf
for ep in range(1, episodes+1):
	env_info = env.reset(train_mode=True)[brain_name]
	states = env_info.vector_observations

	# state = env.reset()
	agent.reset()  # Resets the noise in the agent
	scores = np.zeros(num_agents)
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
	# scores.append(scores)
	print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(ep, np.mean(scores_deque), np.mean(scores)), end="")
	if ep % 100 == 0:
		torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
		torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
		print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque))) 

# Load in the ddpg agent

'''
env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
states = env_info.vector_observations                  # get the current state (for each agent)
scores = np.zeros(num_agents)                          # initialize the score (for each agent)
while True:
    actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
    actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
    env_info = env.step(actions)[brain_name]           # send all actions to tne environment
    next_states = env_info.vector_observations         # get next state (for each agent)
    rewards = env_info.rewards                         # get reward (for each agent)
    dones = env_info.local_done                        # see if episode finished
    scores += env_info.rewards                         # update the score (for each agent)
    states = next_states                               # roll over states to next time step
    if np.any(dones):                                  # exit loop if episode finished
        break
print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
'''

env.close()