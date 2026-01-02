import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import pandas as pd
import yfinance as yf

from src.simulation.monte_carlo import simulate_gbm
from src.simulation.volatility_sources import (
    garch_from_returns,
    ml_from_returns,
    lstm_from_returns,
)

# ======================================================
# Page Config
# ======================================================
st.set_page_config(
    page_title="Indian Market Risk Simulator",
    layout="wide",
)

st.title("Indian Market Monte Carlo Risk Simulator")
st.caption("Dynamic, model-driven risk analysis using GARCH, ML & LSTM")

# ======================================================
# Load Asset Universe
# ======================================================
UNIVERSE_PATH = Path("data/universe/stocks.yaml")

@st.cache_data
def load_assets():
    with open(UNIVERSE_PATH, "r") as f:
        universe = yaml.safe_load(f)

    assets = {}
    for item in universe.get("index", []):
        assets[item["name"]] = item["ticker"]
    for stock in universe.get("stocks", []):
        assets[stock["name"]] = stock["ticker"]
    return assets

ASSETS = load_assets()

# ======================================================
# Yahoo Finance Fetch
# ======================================================
@st.cache_data
def fetch_latest_data(yf_ticker: str):
    df = yf.download(
        yf_ticker,
        start="2014-01-01",
        progress=False,
        auto_adjust=False,
    ).reset_index()

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in col if x]) for col in df.columns]

    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    close_cols = [c for c in df.columns if "close" in c]
    close_col = [c for c in close_cols if "adj" in c]
    close_col = close_col[0] if close_col else close_cols[0]

    df["close"] = df[close_col]
    return df

# ======================================================
# Sidebar Controls
# ======================================================
st.sidebar.header("Simulation Settings")

asset_name = st.sidebar.selectbox("Select Asset", list(ASSETS.keys()))
mode = st.sidebar.radio("Mode", ["Single Model", "Compare Models"])

MODEL_NAMES = [
    "GARCH (Econometric)",
    "ML (Recent Volatility)",
    "LSTM (DL Proxy)",
]

MODEL_COLORS = {
    "GARCH (Econometric)": "#1f77b4",
    "ML (Recent Volatility)": "#ff7f0e",
    "LSTM (DL Proxy)": "#2ca02c",
}

if mode == "Single Model":
    selected_models = [st.sidebar.selectbox("Volatility Model", MODEL_NAMES)]
else:
    selected_models = MODEL_NAMES

confidence = st.sidebar.slider("VaR Confidence Level", 0.90, 0.99, 0.95, 0.01)
n_days = st.sidebar.slider("Forecast Horizon (days)", 1, 60, 30)

run = st.sidebar.button("▶ Run Simulation")

# ======================================================
# Sidebar Explanations
# ======================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Explanations")

with st.sidebar.expander("GARCH (Econometric)"):
    st.write(
        "Captures volatility clustering and mean reversion. "
        "Conservative and stable, but slow to react to regime changes."
    )

with st.sidebar.expander("ML (Recent Volatility)"):
    st.write(
        "Uses recent realized volatility. "
        "Highly reactive, but sensitive to short-term noise."
    )

with st.sidebar.expander("LSTM (Deep Learning)"):
    st.write(
        "Learns non-linear temporal dependencies. "
        "Better at capturing prolonged stress and tail risk."
    )

# ======================================================
# Main Logic
# ======================================================
if run:
    internal_ticker = ASSETS[asset_name]
    yf_ticker = internal_ticker.replace("_", ".")
    if yf_ticker.startswith("NSEI"):
        yf_ticker = "^NSEI"

    df = fetch_latest_data(yf_ticker)
    if df.empty:
        st.error("No market data available.")
        st.stop()

    prices = df["close"]
    returns_hist = np.log(prices / prices.shift(1)).dropna()

    S0 = float(prices.iloc[-1])
    last_date = pd.to_datetime(df["date"].iloc[-1]).date()
    st.success(f"Using market data up to {last_date}")

    results = {}

    for model_name in selected_models:

        if model_name == "GARCH (Econometric)":
            sigma = garch_from_returns(returns_hist)
        elif model_name == "ML (Recent Volatility)":
            sigma = ml_from_returns(returns_hist)
        else:
            sigma = lstm_from_returns(returns_hist)

        paths = simulate_gbm(S0=S0, sigma=sigma, n_days=n_days, n_paths=10_000)

        sim_returns = (paths[-1] - S0) / S0
        var = np.percentile(sim_returns, (1 - confidence) * 100)
        cvar = sim_returns[sim_returns <= var].mean()

        results[model_name] = {
            "sigma": sigma,
            "paths": paths,
            "returns": sim_returns,
            "VaR": var,
            "CVaR": cvar,
        }

    # ==================================================
    # Risk Metrics
    # ==================================================
    st.subheader("Risk Metrics")

    for name, r in results.items():
        st.markdown(f"### {name}")
        c1, c2, c3 = st.columns(3)
        c1.metric("σ Volatility", f"{r['sigma']:.4f}")
        c2.metric(f"{int(confidence*100)}% VaR", f"{r['VaR']:.2%}")
        c3.metric("CVaR", f"{r['CVaR']:.2%}")

    # ==================================================
    # Return Distribution
    # ==================================================
    st.subheader("Return Distribution & Tail Risk")

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, r in results.items():
        color = MODEL_COLORS[name]
        ax.hist(r["returns"], bins=120, density=True, alpha=0.35, color=color, label=name)
        ax.axvline(r["VaR"], linestyle="--", linewidth=2, color=color)

    ax.set_title(f"{asset_name} – {n_days}-Day Return Distribution")
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.legend()
    st.pyplot(fig)

    # ==================================================
    # Monte Carlo Fan Charts
    # ==================================================
    st.subheader("Monte Carlo Price Paths")

    for name, r in results.items():
        paths = r["paths"]
        color = MODEL_COLORS[name]

        p5, p25, p50, p75, p95 = np.percentile(paths, [5, 25, 50, 75, 95], axis=1)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(paths[:, :120], color=color, alpha=0.25, linewidth=0.9)
        ax.fill_between(range(len(p5)), p5, p95, color=color, alpha=0.4, label="5–95%")
        ax.fill_between(range(len(p25)), p25, p75, color=color, alpha=0.5, label="25–75%")
        ax.plot(p50, color=color, linewidth=3.2, label="Median")

        ax.set_title(f"Monte Carlo Paths – {name}")
        ax.set_xlabel("Days")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

    # ==================================================
    # Monte Carlo Explanation
    # ==================================================
    st.markdown("""
### How to read the Monte Carlo simulations

These charts show **thousands of possible future price paths** generated using
the volatility estimated by each model.

- **Median line:** Most likely trajectory  
- **25–75% band:** Normal uncertainty  
- **5–95% band:** Tail risk (extreme scenarios)  

Wider bands imply **higher uncertainty and risk**.
Different models produce different spreads because they encode **different beliefs
about how volatility behaves**.
""")

    # ==================================================
    # Interpretation
    # ==================================================
    st.subheader("Interpretation")

    for name, r in results.items():
        st.info(
            f"""
**Interpretation – {name}**  
With **{int(confidence*100)}% confidence**, losses are not expected to exceed
**{r['VaR']:.2%}** over the next **{n_days} days**.
If losses exceed this level, the **average loss** is around
**{r['CVaR']:.2%}**.
"""
        )

else:
    st.info("Select an asset, choose a mode, and click **Run Simulation**.")
