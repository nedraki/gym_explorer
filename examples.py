# Environments

"""Here’s a bare minimum example of getting something running. 
This will run an instance of the CartPole-v0 environment for 
1000 timesteps, rendering the environment at each step. 
You should see a window pop up rendering the classic cart-pole 
problem: """

import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(100):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()


# Input for continue running code:

x = input('Continue with next example ?')


"""If we ever want to do better than take random actions at 
each step, it’d probably be good to actually know what our 
actions are doing to the environment.

env.step() --> returns:
observation (object)
reward(float)
done(boolean)
info(dict)
"""

"""The process gets started by calling reset(), which returns 
an initial observation. So a more proper way of writing the 
previous code would be to respect the done flag: """

import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()

# Input for continue running code:

x = input('Continue with next example ?:\n')

import gym
env = gym.make('CartPole-v0')
print('action_space', env.action_space)

# Size of action_space ---> Number of actions the agent can take
print('action_space', env.action_space.n)

#> Discrete(2)
print('observation_space', env.observation_space)

# Shape of observation_space
print('observation_space', type(env.observation_space.shape[0]))


#> Box(4,)

print("""\nThe `Discrete` space allows a fixed range of non-negative 
numbers, so in this case valid actions are either 0 or 1. 
The `Box` space represents an n-dimensional box, so valid 
observations will be an array of 4 numbers. 
We can also check the Box’s bounds:""")

print('\nenv.observation_space.high:\n')

print(env.observation_space.high)

print('\nenv.observation_space.low:\n')

print(env.observation_space.low)


####

from gym import spaces
space = spaces.Discrete(8) # Set with 8 elements {0, 1, 2, ..., 7}
x = space.sample()

print('Space:',space)
print('Space sample:', x) 
assert space.contains(x)
assert space.n == 8

"""For CartPole-v0 one of the actions applies force to the left, 
and one of them applies force to the right. 
(Can you figure out which is which?)
Fortunately, the better your learning algorithm, 
the less you’ll have to try to interpret these numbers 
yourself."""