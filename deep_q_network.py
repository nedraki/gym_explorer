import tensorflow as tf
from tensorflow import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam, RMSprop


# Neural Network model for Deep Q Learning

"""Deep Q-Learning
As an agent takes actions and moves through an environment,
it learns to map the observed state of the environment
to an action.
An agent will choose an action in a given state based on
a "Q-value",
which is a weighted reward based on the expected highest
long-term reward.

A Q-Learning Agent learns to perform its task such that the
recommended  action maximizes the potential future rewards.
This method is considered an "Off-Policy" method, meaning its
Q values are updated assuming that the best action was chosen,
even if the best action was not chosen."""


def create_q_model(input_shape: int, num_actions: int):
    """A model of Dense layers (fully connected layers)
    to train an agent using reinforcement learning.

    input_shape: Comes from observation of the environment.
    `env.observation_space.shape[0]`

    num_actions: Number of available actions that can be
    taken by the agent.
    `env.action_space.n`

    For more specific NN, you can find research of Deep Mind research at:
    [docs](https://keras.io/examples/rl/deep_q_network_breakout/)

    """

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(input_shape,)))
    #model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    #model.add(Dropout(0.1))
    model.add(Dense(num_actions, activation='softmax'))

    # model = Model(inputs=X_input, outputs=model, name='trained_model')

    # In the Deepmind paper they use RMSProp however then Adam optimizer
    # improves training time

    model.compile(loss="mse", optimizer=Adam(
        learning_rate=0.001, epsilon=1e-07), metrics=["accuracy"])

    model.summary()

    return model
