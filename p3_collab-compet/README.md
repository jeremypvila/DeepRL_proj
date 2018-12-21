## Deep RL Udacity Course Collaboration and Competition Project

## Project Details

In this repo, I provide my implemntnation of the multi agent deep deterministic policy gradients (MADDPG) algorithm to solve the collaboration and control project in Udacity's Deep Reinforcement (RL) Learning Nano-degree.
This project used the Unity "Tennis" environment.
The trained MADDPG model achieves the objective of the project two agents to play a game of virtual tennis. 
A reward of +0.1 is provided to an agent if it hits the ball over the net.
If an agent lets the ball hit the floor, their side of the table, or hits the ball out of play it gets a score of -0.01.
The project is considered complete if an average score of +0.5 is achieved over a window of 100 episodes (after a maximum is taken between the scores of the agents).
Therefor the goal is to train a pair of agents to play with each other as long as possible.

For each agent, there are 8 variables in the observation space corresponding to position and velocity of the ball and racket. 
For each agent, there are 2 actions, corresponding to moving towards the net and jumping. 
These actions are bounded between -1 and 1.

## Getting Started

Before running any code, please set up an environment (named *drlnd*) with the dependencies listed in: [Deep RL ND](https://github.com/udacity/deep-reinforcement-learning#dependencies).
This environment contains the Unity-ML, as well as openAI gym, pytorch, etc.

Next, download the specific unity environment for your OS using the links below.
For the following environments, download and extract in the "p2_continuous_control" directory:
* Linux (with Vis) - [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
* Mac OSX - [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
* Windows 32 bit - [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
* Windows 64 bit - [https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

If you need to download your own environment, change the path in train.py in the line 

env = UnityEnvironment(file_name="path_to_file")

to reflect the correct path to the environment file.

## Instructions

To train, please load the *drlnd* conda environtment that was set up in "Getting Started."  

### Training

To train from scratch, simply load the conda environment and invoke "python train.py" that outputs:
- Trained model weights in a file called "cc_actor.pth" and "cc_critic.pth" in the base directory.
- Plot of all scores in a file called "trained_scores.png" in the base directory.

To load weights into the agent, simply call the ".load_weights()" method in the train.py script.  
In this method, the path to the weight file is defaulted to the provided trained weights, but can be specified to another file.

Coded by Jeremy Vila
12/21/18
