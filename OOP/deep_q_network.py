import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym

# Neural Network model for Deep Q Learning

"""Deep Q-Learning

As an agent takes actions and moves through an environment, 
it learns to map the observed state of the environment to an action. 
An agent will choose an action in a given state based on a "Q-value", 
which is a weighted reward based on the expected highest long-term reward.

A Q-Learning Agent learns to perform its task such that the recommended 
action maximizes the potential future rewards. This method is considered an
 "Off-Policy" method, meaning its Q values are updated assuming that the best 
 action was chosen, even if the best action was not chosen."""

def create_q_model(input_shape:int, num_actions:int):

    """A model of Dense layers (fully connected layers)
    to train an agent using reinforcement learning. """

    """ 
    input_shape: Comes from observation of the environment.
    `env.observation_space.shape[0]`

    num_actions: Number of available actions that can be
    taken by the agent.
    `env.action_space.n`
    
    For more specific NN, you can find research of Deep Mind research at:
    [docs](https://keras.io/examples/rl/deep_q_network_breakout/)
    
    """
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    model = Model(inputs = X_input, outputs = X, name='trained_model')

    # In the Deepmind paper they use RMSProp however then Adam optimizer
    # improves training time

    model.compile(loss="mse", optimizer=Adam(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    model.summary()


    return model



###################################################################################

class Agent:

    """Starts the execution of environment from gym
    agent_name: Name of environment to play"""

    def __init__(self, agent_name:str):

        self.env = gym.make(agent_name)
        self.num_actions = self.env.action_space.n
        self.input_shape = self.env.observation_space.shape[0]
        self.max_steps_per_episode = 1000

        """Episodes must be defined according with the needs
        for training the Agent"""
        
        self.episodes = 500

    def __str__(self):
        return f"Actions: {self.num_actions} \nShape:{self.input_shape}\nEpisodes:{self.episodes}"

    def run(self):

        for i_episode in range(self.episodes):

            #Init observations with each episode
            observation = self.env.reset()

            # Open cycle of timesteps, 
            # One action complete one timestep, 

            for timestep in range(1, max_steps_per_episode):
                
                #Display the environment
                self.env.render()

                #Decide an action
                action = self.env.action_space.sample()

                # Taking an actions and getting metrics:
                observation, reward, done, info = self.env.step(action)
                if done: # Done is a bolean when True, the objective is achieved.
                    print(f"Episode finished after {t+1} timesteps")
                #Stop the program if the objective is achieved
                    break
        self.env.close()


if __name__ == "__main__":
    agent = Agent('CartPole-v1')
    agent.run()