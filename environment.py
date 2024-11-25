import numpy as np
import gym
from gym import spaces


class OnlineMidPriceForecastingEnv(gym.Env):
    """
    Custom OpenAI Gym environment for forecasting mid-prices
    in a limit order book.
    """
    def __init__(self, initial_data, window_size,
                 action_lower_bound, action_upper_bound):
        """
        Initializes the environment.

        Parameters:
        - initial_data (numpy.ndarray): Input data for the environment.
        - window_size (int): Size of the rolling window for observations.
        - action_lower_bound (float): Lower bound for the action space.
        - action_upper_bound (float): Upper bound for the action space.
        """
        super().__init__()
        self.data = initial_data
        self.window_size = window_size
        self.current_step = 0

        self.action_space = spaces.Box(
            low=np.array([action_lower_bound]),
            high=np.array([action_upper_bound]),
            dtype=np.float64
        )
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(initial_data.shape[1] - 1,),
            dtype=np.float64
        )

    def reset(self):
        """
        Resets the environment to its initial state.

        Returns:
        - numpy.ndarray: Initial state of the environment.
        """
        self.current_step = max(0, len(self.data) - self.window_size)
        return self.data[self.current_step, :-1]

    def get_mid_price(self):
        """
        Retrieves the current mid-price from the environment.

        Returns:
        - float: Current mid-price.
        """
        return self.data[self.current_step, -1]

    def step(self, action, epsilon):
        """
        Executes the agent's action in the environment.

        Parameters:
        - action (float): Action taken by the agent.
        - epsilon (float): Exploration factor for reward calculation.

        Returns:
        - next_state (numpy.ndarray): The next state of the environment.
        - reward (float): Reward for the action taken.
        - done (bool): Whether the episode has ended.
        - true_mid_price (float): The actual mid-price at the current step.
        """
        done = self.current_step >= len(self.data)
        true_mid_price = np.around(
            self.data[self.current_step, -1]
            if not done else self.data[self.current_step - 1, -1],
            decimals=2
        )
        reward = -abs(action - true_mid_price) * (1 - epsilon)

        self.current_step += 1

        next_state = (
            self.data[self.current_step - 1, :-1]
            if not done else np.zeros(self.data.shape[1] - 1)
        )
        return next_state, reward, done, true_mid_price
