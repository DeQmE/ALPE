import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import expon, poisson

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def kernel_features(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Generate kernelized features for LOB data.

    Applies non-linear transformations and cross-products to bid/ask prices.

    Args:
        features: Array (n_samples, 4) with bid/ask prices and volumes
                  (order: bid_price, bid_volume, ask_price, ask_volume).
        labels: Array (n_samples,) with mid-prices.

    Returns:
        Array (n_samples, 12) with original, kernelized, and mid-price features.
    """
    n_samples = features.shape[0]
    bid_ask = features[:, [0, 2]]  # Bid price, ask price
    kernel = np.zeros((n_samples, 3))
    kernel[:, 0] = np.diag(np.dot(bid_ask, bid_ask.T))
    kernel[:, 1] = (1 + kernel[:, 0]) ** 2
    kernel[:, 2] = (1 + kernel[:, 0]) ** 3
    synthesized = np.column_stack((
        bid_ask * bid_ask[:, [1, 0]],  # Cross-products
        bid_ask ** 2                   # Squared terms
    ))
    return np.hstack((features, kernel, synthesized, labels.reshape(-1, 1)))

def generate_gbm_path(
    initial_price: float,
    drift: float,
    volatility: float,
    n_ticks: int,
    tick_rate: float
) -> np.ndarray:
    """Generate mid-price path using Geometric Brownian Motion.

    Uses exponential inter-tick times for HFT-like irregular arrivals.

    Args:
        initial_price: Starting price.
        drift: Annual drift rate (mu).
        volatility: Annual volatility (sigma).
        n_ticks: Number of ticks to generate.
        tick_rate: Ticks per second (lambda).

    Returns:
        Array (n_ticks,) with mid-prices.
    """
    prices = np.zeros(n_ticks)
    prices[0] = initial_price
    for i in range(1, n_ticks):
        dt = expon.rvs(scale=1/tick_rate) / 86400  # Seconds to days
        drift_term = drift * prices[i-1] * dt
        diffusion = volatility * prices[i-1] * np.random.normal(0, np.sqrt(dt))
        prices[i] = max(prices[i-1] + drift_term + diffusion, 0.01)
    return prices

def generate_synthetic_lob(
    initial_price: float = 123.45,
    drift: float = 0.05,
    volatility: float = 0.25,
    n_samples: int = 100000,
    tick_rate: float = 3.0,
    volume_rate: float = 1000.0,
    spread_pct: float = 0.0005,
    stock_id: int = 1,
    output_dir: str = 'data/synthetic'
) -> None:
    """Generate synthetic Limit Order Book (LOB) data for a single stock.

    Simulates HFT LOB data with bid/ask prices, volumes, and kernelized features.
    Uses generic parameters for reproducibility, not real market data.

    Disclaimer: Parameters (e.g., initial_price, drift, volatility) are illustrative
    and chosen for demonstration. They do not reflect proprietary data or specific
    market conditions used in the original study.

    Args:
        initial_price: Starting mid-price.
        drift: Annual drift rate.
        volatility: Annual volatility.
        n_samples: Number of ticks.
        tick_rate: Ticks per second.
        volume_rate: Poisson rate for volumes.
        spread_pct: Bid-ask spread percentage.
        stock_id: Stock identifier for naming.
        output_dir: Output directory for CSV.

    Raises:
        ValueError: If inputs are invalid.
    """
    if any(x <= 0 for x in [initial_price, n_samples, tick_rate, volume_rate]):
        raise ValueError("Parameters must be positive.")
    
    logger.info(f"Generating synthetic LOB data for stock_{stock_id}")
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42 + stock_id)
    
    timestamps = [pd.Timestamp('2022-01-01 09:30:00.000')]
    for _ in range(n_samples - 1):
        dt_ms = expon.rvs(scale=1/tick_rate) * 1000
        timestamps.append(timestamps[-1] + pd.Timedelta(milliseconds=dt_ms))
    
    mid_prices = generate_gbm_path(
        initial_price, drift, volatility, n_samples, tick_rate
    )
    
    bid_prices = mid_prices * (1 - spread_pct / 2)
    ask_prices = mid_prices * (1 + spread_pct / 2)
    bid_volumes = poisson.rvs(volume_rate, size=n_samples)
    ask_volumes = poisson.rvs(volume_rate, size=n_samples)
    bid_volumes[bid_volumes < 1] = 1
    ask_volumes[ask_volumes < 1] = 1
    
    features = np.column_stack((
        bid_prices, bid_volumes, ask_prices, ask_volumes
    ))
    X_combined = kernel_features(features, mid_prices)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'bid_price': X_combined[:, 0],
        'bid_volume': X_combined[:, 1].astype(int),
        'ask_price': X_combined[:, 2],
        'ask_volume': X_combined[:, 3].astype(int),
        'kernel_linear': X_combined[:, 4],
        'kernel_quadratic': X_combined[:, 5],
        'kernel_cubic': X_combined[:, 6],
        'cross_bid_ask': X_combined[:, 7],
        'cross_ask_bid': X_combined[:, 8],
        'bid_squared': X_combined[:, 9],
        'ask_squared': X_combined[:, 10],
        'mid_price': X_combined[:, 11]
    })
    
    output_file = os.path.join(output_dir, f'synthetic_stock_{stock_id}.csv')
    df.to_csv(output_file, index=False)
    logger.info(f"Saved: {output_file} with {n_samples} samples")

def main():
    """Generate synthetic LOB data for 5 stocks."""
    for stock_id in range(1, 6):
        try:
            generate_synthetic_lob(stock_id=stock_id)
        except Exception as e:
            logger.error(f"Failed to generate data for stock_{stock_id}: {e}")

if __name__ == '__main__':
    main()
