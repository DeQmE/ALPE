# ALPE Agent for Mid-Price Forecasting in High-Frequency Trading

This repository contains the implementation of the Adaptive Learning Policy Engine (ALPE) for mid-price forecasting in high-frequency trading (HFT). 

## Description
The ALPE framework is designed to:
- Predict mid-prices in high-frequency trading using limit order book data.
- Operate in an online, event-driven manner, learning and adapting dynamically without batch updates.
- Demonstrate robust performance using evaluation metrics such as MSE, RMSE and the novel RRMSE score.


## Installation
Prerequisites: Python 3.8 or higher should be installed on your system.
Install the required libraries using the requirements.txt file.

## Training the ALPE Agent
To train and evaluate the ALPE agent, use the following command:  
python main.py --epochs  1

## Outputs
Training Logs: The console will display metrics such as MSE and RMSE during training.  
Predictions: The agent will predict and compare mid-prices step-by-step.

## Code Structure

Here is an overview of the key files in this repository:  

```
├── main.py                     # Main script to train and evaluate the ALPE agent  
├── environment.py              # Custom environment for mid-price forecasting  
├── agent.py                    # ALPE agent implementation  
├── utils.py                    # Utility functions (data generation, preprocessing, evaluation)  
├── README.md                   # Documentation (this file)  
├── requirements.txt            # List of dependencies for the project  
```


