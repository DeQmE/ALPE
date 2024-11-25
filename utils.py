import numpy as np
import pandas as pd


def generate_synthetic_lob(num_rows=30, base_price=100):
    """
    Generates synthetic limit order book data for demonstration purposes only

    Parameters:
    - num_rows (int): Number of rows for the synthetic LOB.
    - base_price (float): Base price for the LOB data.

    Returns:
    - pd.DataFrame: A DataFrame containing synthetic LOB data.
    """
    ask_prices = np.around(
        base_price + np.random.uniform(0.01, 0.5, num_rows),
        decimals=2
    )
    bid_prices = np.around(
        base_price - np.random.uniform(0.01, 0.5, num_rows),
        decimals=2
    )
    ask_volumes = np.random.randint(10, 101, num_rows)
    bid_volumes = np.random.randint(10, 101, num_rows)

    df = pd.DataFrame({
        'Ask Price': ask_prices,
        'Ask Volume': ask_volumes,
        'Bid Price': bid_prices,
        'Bid Volume': bid_volumes
    })
    df['Mid Price'] = (df['Ask Price'] + df['Bid Price']) / 2
    df['Mid Price (Label)'] = df['Mid Price'].shift(-1)

    return df.iloc[:-1]


def preprocess_lob(lob):
    """
    Preprocesses the limit order book for training.

    Parameters:
    - lob (pd.DataFrame): The LOB data.

    Returns:
    - tuple: (input_data, labels) where:
        - input_data (pd.DataFrame): Feature columns.
        - labels (pd.Series): Target column (next mid-price).
    """
    input_data = lob.iloc[:, :-1]
    labels = lob.iloc[:, -1]
    return input_data, labels


def calculate_mse(predictions, true_values):
    """
    Calculates Mean Squared Error (MSE).

    Parameters:
    - predictions (numpy.ndarray): Predicted values.
    - true_values (numpy.ndarray): Actual values.

    Returns:
    - float: The MSE value.
    """
    return np.mean((predictions - true_values) ** 2)


def calculate_rmse(predictions, true_values):
    """
    Calculates Root Mean Squared Error (RMSE).

    Parameters:
    - predictions (numpy.ndarray): Predicted values.
    - true_values (numpy.ndarray): Actual values.

    Returns:
    - float: The RMSE value.
    """
    return np.sqrt(calculate_mse(predictions, true_values))


def exe_alpe_agent(env, agent, training_data):
    """
    Executes the ALPE agent within the given environment.

    Parameters:
    - env (gym.Env): The environment for the agent.
    - agent (object): The ALPE agent.
    - training_data (numpy.ndarray): Training data for the environment.

    Returns:
    - tuple: (rl_predictions_train, rl_true_values_train) where:
        - rl_predictions_train (list): Predicted mid-prices.
        - rl_true_values_train (list): Actual mid-prices.
    """
    state_train = env.reset()
    current_mid_price = env.get_mid_price()
    rl_predictions_train, rl_true_values_train = [], []

    done = False
    while not done:
        action_train = agent.act(state_train, current_mid_price)
        action_train = np.around(action_train, decimals=2)

        next_state, reward, done, true_mid_price = env.step(
            action_train, agent.epsilon
        )
        agent.train(state_train, action_train, reward, next_state, done)

        rl_predictions_train.append(action_train)
        rl_true_values_train.append(true_mid_price)

        state_train = next_state

    return rl_predictions_train, rl_true_values_train
