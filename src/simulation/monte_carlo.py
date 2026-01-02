import numpy as np
import pandas as pd
from pathlib import Path

# -------------------------
# Config
# -------------------------
N_PATHS = 10_000      # number of Monte Carlo simulations
N_DAYS = 30           # forecast horizon (days)
TRADING_DAYS = 252

DATA_DIR = Path("data/raw/multi_asset")


# -------------------------
# GBM Simulation
# -------------------------
def simulate_gbm(
    S0: float,
    sigma: float,
    mu: float = 0.0,
    n_days: int = N_DAYS,
    n_paths: int = N_PATHS,
):
    dt = 1 / TRADING_DAYS

    # Random shocks
    Z = np.random.normal(0, 1, size=(n_days, n_paths))

    # Log returns
    log_returns = (
        (mu - 0.5 * sigma**2) * dt
        + sigma * np.sqrt(dt) * Z
    )

    # Price paths
    price_paths = S0 * np.exp(np.cumsum(log_returns, axis=0))

    return price_paths


# -------------------------
# Load latest price
# -------------------------
def get_latest_price(asset_id: str):
    file = DATA_DIR / f"{asset_id}.csv"
    df = pd.read_csv(file)
    return df.iloc[-1]["close"]


# -------------------------
# Example Run
# -------------------------
def main():
    # Example asset
    asset_id = "NSEI"      # try RELIANCE_NS, HDFCBANK_NS later
    forecast_vol = 0.25    # <-- placeholder (will come from model later)

    S0 = get_latest_price(asset_id)

    paths = simulate_gbm(
        S0=S0,
        sigma=forecast_vol,
    )

    print("Monte Carlo simulation complete")
    print("Paths shape:", paths.shape)
    print("Final price mean:", paths[-1].mean())
    print("Final price std:", paths[-1].std())


if __name__ == "__main__":
    main()
