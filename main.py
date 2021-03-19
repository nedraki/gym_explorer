import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from deep_q_network import create_q_model
from collections import deque
import gym


class Agent_Name:

    """Selection of environment of interest"""

    # Game selector:

    def __init__(self):

        print("1: LunarLander, \n2:CartPole")
        self.user_input = input('Select a game from the list\n')
        if self.user_input == '1':
            self.agent_name = 'LunarLander-v2'
        elif self.user_input == '2':
            self.agent_name = 'CartPole-v0'


class Agent:

    """Starts the execution of environment from gym
    agent_name: Name of environment to play"""

    def __init__(self, agent_name: str):

        self.env = gym.make(agent_name)

        # Hyperparameters

        # This could be moved to and independent file

        # Episodes must be defined according with the needs
        # for training the Agent

        # number of games we want the agent to play.
        self.episodes = 500
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        # Doubly Ended Queue (faster than lists)
        self.memory = deque(maxlen=2000)

        # Decay or discount rate, to calculate the future discounted reward.
        self.gamma = 0.95

        # Exploration rate: the rate in which an agent randomly decides its
        # action rather than prediction.
        self.epsilon = 1.0

        # The agent will explore at least this amount.
        self.epsilon_min = 0.001

        # To decrease the number of explorations as it gets good at playing games.
        self.epsilon_decay = 0.999

        # Determines how much memory DQN will use to learn.
        self.batch_size = 64

        # Maximun steps per episode

        self.maximun_steps = 1000

        # Import and Create NN model:

        self.model = create_q_model(self.state_size, self.action_size)

    def __str__(self):
        return f"Actions: {self.action_size} \nShape:{self.state_size}\nEpisodes:{self.episodes}"

    def remember_experiences(self, state, action, reward, next_state, done):
        """ To keep track of the previous decisions taken by the agent,
        and re-train the neural network with it. As humans, the NN tends
        to forget previous experiences, in those cases of negative reward,
        we don't want to stumble over the same stone """

        self.memory.append((state, action, reward, next_state, done))

        # Verify if the agent has achieved enough experience:

        if len(self.memory) > self.maximun_steps:
            if self.epsilon > self.epsilon_min:
                # The agent progresively takes less random actions
                self.epsilon *= self.epsilon_decay

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            # Returns a random action for the agent
            return random.randrange(self.action_size)
        else:
            # returns the max Q value as an action
            return np.argmax(self.model.predict(state))

    def replay(self):
        """ A method to train the NN with the experiences saved on memory"""

        # Random selection of experiences from memory

        if len(self.memory) < self.maximun_steps:
            return None
        else:
            mini_batch = random.sample(self.memory, min(
                len(self.memory), self.batch_size))

            state = np.zeros((self.batch_size, self.state_size))

            next_state = np.zeros((self.batch_size, self.state_size))

            action, reward, done = [], [], []

            # Before prediction:
            # The operation can be improved using tensors

            for i in range(self.batch_size):

                state[i] = mini_batch[i][0]
                action.append(mini_batch[i][1])
                reward.append(mini_batch[i][2])
                next_state[i] = mini_batch[i][3]
                done.append(mini_batch[i][4])

            # Batch predictions (this save computing time but...how much ?)
            # Predictions
            target = self.model.predict(state)
            target_next = self.model.predict(next_state)

            for i in range(self.batch_size):

                if done[i]:
                    target[i][action[i]] = reward[i]
                else:

                    # Application of DEEP Q Network (DQN)
                    # We can express the target in a magical one line of code
                    # in python:

                    target[i][action[i]] = reward[i] + \
                        self.gamma * (np.amax(target_next[i]))

            # Train NN with batches
            # verbose shows the training progress (0:silent,1:progress bar, 2:epoch)
            self.model.fit(
                state, target, batch_size=self.batch_size, verbose=0)

    def save_model(self, agent_name):
        self.model.save(self.agent_name)

    def run(self):

        for i_episode in range(self.episodes):

            # Init observations with each episode
            state = self.env.reset()
            # Check the first output of state and
            # its original shape
            state = np.reshape(state, [1, self.state_size])
            # boolean stating whether the environment is terminated
            done = False
            i = 0

            while done == False:
                # Display the gym environment ###############
                #self.env.render()
                # Decide an actions
                action = self.take_action(state)
                # Execute action and observe results:
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])

                if done == False or i == self.env._max_episode_steps - 1:
                    reward = reward
                else:
                    reward = -100

                self.remember_experiences(
                    state, action, reward, next_state, done)
                state = next_state
                i += 1
                # Once the environment is terminated:
                if done:
                    print("episode: {}, score: {}".format(i_episode, i))


                    if i == self.env._max_episode_steps:
                        print(f'Saving trained model {agent_name}')
                        self.save_model(f'trained_{agent_name}.h5')

                self.replay()


if __name__ == "__main__":
    game = Agent_Name()
    agent = Agent(game.agent_name)
    agent.run()
