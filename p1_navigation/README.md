# Deep RL Navigation Project Implementation

## Project Details

The goal of the Deep RL Navigation project is to train and agent navigate a square environment and collect as many yellow bananas and avoid as many blue bananas as possible.
Specifically, the agent receives a reqard of +1 for a yellow banana, and -1 for a blue banana.  
The agent is considered trained after getting an average score of +13 over 100 consecutive episodes.

At any given time t, the agent can take one of four discrete actions:
* 0 - move forward
* 1 - move backward
* 2 - turn left
* 3 - turn right

Additionally, at every time t, the agent receives state information from its environment.
The state space is continuous of dimension 37, that contains the agent's velocity and other ray based perception of objects in it's forward direction.

## Getting Started

Before running any code, please set up an environment (named *drlnd*) with the dependencies listed in: [Deep RL ND](https://github.com/udacity/deep-reinforcement-learning#dependencies).
This environment contains the Unity-ML, as well as openAI gym, pytorch, etc.

Next, download the specific unity environment for your OS using the links below.
For the following environments, download and extract in the "p1_navigation" directory:
* Linux (with Vis) - [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* Mac OSX - [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* Windows 32 bit - [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* Windows 64 bit - [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

If you need to download your own environment, change the path in train.py in the line 

env = UnityEnvironment(file_name="path_to_file")

to reflect the correct path to the environment file.


## Instructions

To train, please load the *drlnd* conda environtment that was set up in "Getting Started."  

### Training

To train from scratch, simply load the conda environment and invoke "python train.py" that outputs:
- Trained model weights in a file called "navigation.pth" in the base directory.
- Plot of all scores in a file called "trained_scores.png" in the base directory.

To load weights into the agent, simply call the ".load_weights()" method.  In this method, the path to the weight file is defaulted to the provided trained weights, but can be specified to another file.

Coded by Jeremy Vila
