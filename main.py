#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main file to train the ALPE agent for mid-price forecasting.
"""

# Standard library imports
import logging
import argparse

import numpy as np

# Project-specific imports
from environment import OnlineMidPriceForecastingEnv
from agent import ALPEagent
from utils import generate_synthetic_lob, preprocess_lob, calculate_mse
from utils import calculate_rmse, exe_alpe_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def test_and_evaluate(epochs):
    """
    Trains the ALPE agent and evaluates its performance.

    Parameters:
    epochs (int): The number of epochs for training.
    """
    logging.info("Generating synthetic limit order book data...")
    synthetic_lob = generate_synthetic_lob()

    logging.info("Preprocessing data...")
    input_data, labels = preprocess_lob(synthetic_lob)

    training_data = np.hstack((
        input_data.iloc[:, :-1],
        labels.to_numpy().reshape(-1, 1)
    ))

    logging.info("Initializing environment and agent...")
    env = OnlineMidPriceForecastingEnv(
        training_data,
        window_size=1,
        action_lower_bound=-0.1,
        action_upper_bound=0.1,
    )
    agent = ALPEagent(
        state_size=env.observation_space.shape[0],
        action_space=env.action_space,
    )

    logging.info(f"Training ALPE agent for {epochs} epochs...")
    rl_predictions, rl_true_values = exe_alpe_agent(
        env, agent, training_data
    )

    # Evaluate model performance
    mse = calculate_mse(
        np.array(rl_predictions),
        np.array(rl_true_values)
    )
    rmse = calculate_rmse(
        np.array(rl_predictions),
        np.array(rl_true_values)
    )

    logging.info(f"Training completed. MSE: {mse}, RMSE: {rmse}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train the ALPE agent for mid-price forecasting."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)",
    )
    args = parser.parse_args()

    # Execute testing and evaluation
    test_and_evaluate(args.epochs)
