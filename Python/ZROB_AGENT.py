from os import path
import numpy as np
import tensorflow as tf
from keras import layers
import numpy as np


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=0.01, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer:
    def __init__(self, buffer_capacity=1000000, batch_size=256, num_states=1, num_actions=1):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,actor_model, critic_model,target_actor, target_critic, actor_optimizer, critic_optimizer, gamma
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self, actor_model, critic_model,target_actor, target_critic, actor_optimizer, critic_optimizer, gamma):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch,actor_model, critic_model,target_actor, target_critic, actor_optimizer, critic_optimizer, gamma)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))



def get_actor(num_states=1, num_actions=1, upper_bound=[]):
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003, seed=None)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(150, activation="relu",
                       kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1/np.sqrt(num_states), maxval=1/np.sqrt(num_states), seed=None),# type: ignore
                       kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))(inputs)# type: ignore
    out1 = layers.Dense(100, activation="relu",
                        kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1/np.sqrt(400), maxval=1/np.sqrt(400), seed=None),# type: ignore
                        kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))(out)# type: ignore
    outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init,# type: ignore
                           kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))(out1) # type: ignore

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic(num_states=1, num_actions=1):
    last_init = tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003, seed=None)
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(20, activation="relu",
                             kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1/np.sqrt(num_states), maxval=1/np.sqrt(num_states), seed=None),# type: ignore
                             kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))(state_input)# type: ignore
    state_out1 = layers.Dense(30, activation="relu",
                              kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1/np.sqrt(20), maxval=1/np.sqrt(20), seed=None),# type: ignore
                              kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))(state_out)# type: ignore

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(40, activation="relu",
                              kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1/np.sqrt(num_actions), maxval=1/np.sqrt(num_actions), seed=None),# type: ignore
                              kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))(action_input)# type: ignore# type: ignore

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out1, action_out])

    out = layers.Dense(150, activation="relu",
                       kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1/np.sqrt(70), maxval=1/np.sqrt(70), seed=None),# type: ignore
                       kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))(concat)# type: ignore
    out1 = layers.Dense(100, activation="relu",
                        kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1/np.sqrt(400), maxval=1/np.sqrt(400), seed=None),# type: ignore
                        kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))(out)# type: ignore
    outputs = layers.Dense(1, kernel_initializer=last_init,# type: ignore
                           kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))(out1)# type: ignore

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


def get_pred(num_states=1, num_actions=1):
    last_init = tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003, seed=None)
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(20, activation="relu",
                             kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1/np.sqrt(num_states), maxval=1/np.sqrt(num_states), seed=None),# type: ignore
                             kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))(state_input)# type: ignore
    state_out1 = layers.Dense(30, activation="relu",
                              kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1/np.sqrt(20), maxval=1/np.sqrt(20), seed=None),# type: ignore
                              kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))(state_out)# type: ignore

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(40, activation="relu",
                              kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1/np.sqrt(num_actions), maxval=1/np.sqrt(num_actions), seed=None),# type: ignore
                              kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))(action_input)# type: ignore# type: ignore

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out1, action_out])

    out = layers.Dense(200, activation="relu",
                       kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1/np.sqrt(70), maxval=1/np.sqrt(70), seed=None),# type: ignore
                       kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))(concat)# type: ignore
    out1 = layers.Dense(150, activation="relu",
                        kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1/np.sqrt(400), maxval=1/np.sqrt(400), seed=None),# type: ignore
                        kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))(out)# type: ignore
    outputs = layers.Dense(num_states, kernel_initializer=last_init,# type: ignore
                           kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))(out1)# type: ignore

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model



def policy(state, noise_object, actor_model, lower_bound, upper_bound):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return legal_action
