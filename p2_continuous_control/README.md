# # Deep RL Continuous Control Project Implementation

## Project Details

In this project, the goal is to control a robotic arm can move to target locations. 
A reward of +0.1 is provided for each step that the agent's hand is in the goal location. 
Therefore, the goal is to get the hand to the target as quickly as possible, then remain in that location

There are 33 variables in the observation space corresponding to position, rotation, velocity, and angular velocities of the arm. 
There are 4 actions, corresponding to torque applicable to two joints. 
These actions are bounded between -1 and 1.

We apply the deep deterministic policy gradients (ddpg) to solve the single "Reacher" robot example.
We did not consider the distributed training example of controlling many arms in parallel due to computational constraints.

## Getting Started

Before running any code, please set up an environment (named *drlnd*) with the dependencies listed in: [Deep RL ND](https://github.com/udacity/deep-reinforcement-learning#dependencies).
This environment contains the Unity-ML, as well as openAI gym, pytorch, etc.

Next, download the specific unity environment for your OS using the links below.
For the following environments, download and extract in the "p2_continuous_control" directory:
* Linux (with Vis) - [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
* Mac OSX - [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
* Windows 32 bit - [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
* Windows 64 bit - [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

If you need to download your own environment, change the path in train.py in the line 

env = UnityEnvironment(file_name="path_to_file")

to reflect the correct path to the environment file.

## Instructions

To train, please load the *drlnd* conda environtment that was set up in "Getting Started."  

### Training

To train from scratch, simply load the conda environment and invoke "python train.py" that outputs:
- Trained model weights in a file called "cc_actor.pth" and "cc_critic.pth" in the base directory.
- Plot of all scores in a file called "trained_scores.png" in the base directory.

To load weights into the agent, simply call the ".load_weights()" method.  
In this method, the path to the weight file is defaulted to the provided trained weights, but can be specified to another file.

Coded by Jeremy Vila
