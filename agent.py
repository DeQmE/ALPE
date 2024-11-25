
import numpy as np
import tensorflow as tf


class ALPEagent:
    """
    RL ALPE agent for mid-price forecasting in a limit order book.
    """
    def __init__(self, state_size, action_space, epsilon_decay=0.999):
        """
        Initializes the DQN agent.

        Parameters:
        - state_size (int): The size of the state space.
        - action_space (gym.Space): The environment's action space.
        - epsilon_decay (float): Decay rate for the exploration factor.
        """
        self.state_size = state_size
        self.action_space = action_space
        self.learning_rate = 0.0001
        self.epsilon = 0.99
        self.epsilon_min = 0.001
        self.epsilon_decay = epsilon_decay
        self.model = self.build_model()

    def build_model(self):
        """
        Builds the deep ALPE MLP-based policy forecaster.

        Returns:
        - tf.keras.Model: Compiled neural network model.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                16, input_dim=self.state_size, activation="relu"
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1)
        ])
        model.compile(
            loss="mean_squared_error",
            optimizer=tf.keras.optimizers.legacy.Adam(
                learning_rate=self.learning_rate
            )
        )
        return model

    def act(self, state, current_mid_price):
        """
        Selects an action based on the current state.

        Parameters:
        - state (numpy.ndarray): Current state of the environment.
        - current_mid_price (float): Current mid-price.

        Returns:
        - float: Predicted mid-price or exploratory action.
        """
        if np.random.rand() <= self.epsilon:
            random_action = np.random.uniform(
                self.action_space.low[0], self.action_space.high[0]
            )
            return current_mid_price + random_action
        state = np.reshape(state, [1, self.state_size])
        return self.model.predict(state, verbose=0)[0][0]

    def train(self, state, action, reward, next_state, done):
        """
        Trains the agent's neural network.

        Parameters:
        - state (numpy.ndarray): Current state of the environment.
        - action (float): Action taken by the agent.
        - reward (float): Reward received.
        - next_state (numpy.ndarray): The next state observed.
        - done (bool): Whether the episode has ended.
        """
        predicted_q_val = self.model.predict(
            np.reshape(state, [1, self.state_size]),
            verbose=0
        )[0][0]
        target = reward - abs(action - predicted_q_val) * (1 - self.epsilon)

        if not done:
            target += self.epsilon * max(self.epsilon_min, self.epsilon_decay)

        state = np.reshape(state, [1, self.state_size])
        self.model.fit(state, np.array([target]), epochs=20, verbose=0)

    def update_epsilon(self):
        """
        Updates the exploration factor (epsilon) with decay.
        """
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

