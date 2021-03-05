import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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

num_actions = 4

def create_q_model():

    """ num_actions: Number of available actions that can be
    taken by the agent 

    [docs](https://keras.io/examples/rl/deep_q_network_breakout/)
    """

    # Network defined by the Deepmind paper

    inputs = layers.Input(shape=(32, 32, 4,))

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)


# The first model makes the predictions for Q-values which are used to
# make a action.

model = create_q_model()

# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.

model_target = create_q_model()

###################################################################################

