# Getting Started with Gym

The following document is a short handbook to help you get your
hands dirty using OpenAI Gym and Reinforcement Learning. Is meant to be a summary of the official documentation and source code.

The presented code is focused on a development environment
running Python on Ubuntu (Linux distribution).

## About the files

`examples.py` The code it's a intro to the main functions in gym. Runs a classic contron environment (CartPole-v0). 

`game_executer.py` Runs the environment selected by the user. ***In construction***

`lunar_lander.py` Runs the LunarLander-v0 env and agent does ramdom actions.

`neural_network.py` ***In construction***

## Basic installation

1. Create a virtual environment for your installation.

2. To perform a minimal installation of `gym`:

`pip install gym`

You can also clone the environment from the official repository:

`git clone https://github.com/openai/gym.git
cd gym
pip install -e .`

After completing the minimal installation a few environments will be available to run.

- algorithmic
- toy_text
- classic_control (you'll need pyglet to render though)

[Full list of available environments](https://gym.openai.com/envs/#classic_control)

Now, you can run the file `examples.py` to take a look at basics examples of Gym working on the environment `CartPole-v1`.

A detailed explanation can be found at:

[Getting started official docs](https://gym.openai.com/docs/#available-environments)

## Full installation

3. On Ubuntu:

`sudo apt-get install -y libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb ffmpeg curl patchelf libglfw3 libglfw3-dev cmake zlib1g zlib1g-dev swig`

-Decide if you want to install or not MuJoco and continue with the next steps. (See the side note)

#### Side note:

On the full installation of Gym, `MuJoco` is one of the environments included, however, this package is proprietary. You need access to a license to be able to run it but they offer 30 days free trial. I would recommend playing around with other environments, understand properly how Gym works and after claimed the license for Mujoco so you can take the most advantage of it.

[Instructions to install MuJoco](https://github.com/openai/mujoco-py#obtaining-the-binaries-and-license-key)

***I installed MuJoco:***

4. After completing the installation of MuJoco and packages in Ubuntu :

`pip install -e '.`
or
`pip install 'gym[all]'`

This will install all the necessary packages for running all gym environments.

***I prefer to install the environments of my preference***

4. Then run the following commands: 

`pip install <environment_of_your_interest>`
or
`pip install 'gym[environment_of_your_interest]'`

*Some examples:*

`pip install box2d-py` For box2D environment

`pip install 'gym[atari]'` For Atari

Note: I'm still looking for better documentation on this topic.

## Some basic concepts

`Environments`: Within the context of OpenAI Gym, an environment represents the "real world", a place for interaction, the place where actions are taking place. This is what we (humans) can easily see on for example the graphics of a video game.

`Agent`: It's basically your code (algorithm), the agent interacts with the environment throughout actions defined by the algorithm logic. For example, this is the software running on the back-end of a video game.

`Observations and step() function`

The core for building a reinforcement learning algorithm, `step()` returns four values. These are:

- `observation` (object): an environment-specific object representing your observation of the environment. For example, pixel data from a camera, joint angles and joint velocities of a robot, or the board state in a board game.

- `reward` (float): the amount of reward achieved by the previous action. The scale varies between environments, but the goal is always to increase your total reward.

- `done` (boolean): whether it’s time to reset the environment again. Most (but not all) tasks are divided up into well-defined episodes, and done being True indicates the episode has terminated.

- `info` (dict): diagnostic information useful for debugging. It can sometimes be useful for learning (for example, it might contain the raw probabilities behind the environment’s last state change). However, official evaluations of your agent are not allowed to use this for learning.

This is just an implementation of the classic “agent-environment loop”. Each timestep, ***the agent chooses an action, and the environment returns an observation and a reward.***

## Getting specifical info about the envs

The best way is to read the source code corresponding to the environment of interest.

The main information will be written on a docstring with clear indications of values representing the physics of the environment and the `int` that correspond to a specific action.

For example:

[Description CartPoleEnv](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py)


