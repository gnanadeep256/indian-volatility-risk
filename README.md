# Indian Market Monte Carlo Risk Simulator

A **model-driven risk analysis system** for Indian financial markets using  
**Monte Carlo simulation**, **Value at Risk (VaR)**, and **Conditional VaR (CVaR)**  
powered by **GARCH**, **Machine Learning**, and **LSTM-based volatility models**.

The app dynamically fetches the latest market data and visualizes **uncertainty, tail risk, and downside exposure** in an intuitive way.

---

## Key Features

-  **Monte Carlo price simulations** using Geometric Brownian Motion (GBM)
-  **Risk metrics**: Volatility (σ), VaR, CVaR
-  **Dynamic data fetching** (always uses latest market data)
-  **Multiple volatility models**
  - GARCH (econometric)
  - ML (recent volatility)
  - LSTM (deep learning proxy)
-  **Compare Models mode** (side-by-side risk views)
-  **Fan charts** for uncertainty visualization
-  Clear explanations for **how to read every chart**

---

## 🧠 Volatility Models Explained

### 1️ GARCH (Econometric)
- Models volatility clustering and mean reversion
- Stable and conservative
- Slower to react to regime shifts

### 2️ ML (Recent Volatility)
- Uses recent realized volatility patterns
- Highly reactive to market shocks
- Sensitive to short-term noise

### 3️ LSTM (Deep Learning Proxy)
- Learns non-linear temporal dependencies
- Captures prolonged stress and tail risk
- Produces wider uncertainty bands

Each model leads to **different Monte Carlo outcomes** because volatility assumptions differ.

---

##  What the Monte Carlo Simulation Shows

Each simulation generates **thousands of possible future price paths**.

### Chart elements:
- **Thin lines** → Individual simulated futures  
- **Median line** → Most likely price path  
- **25–75% band** → Normal uncertainty  
- **5–95% band** → Extreme but plausible outcomes  

 Wider bands = higher uncertainty and risk.

> Monte Carlo simulations do **not predict the future**.  
They show *what could happen if current volatility assumptions persist*.

---

## Risk Metrics

- **σ (Volatility)**  
  Expected magnitude of price fluctuations

- **Value at Risk (VaR)**  
  Maximum expected loss at a given confidence level

- **Conditional VaR (CVaR)**  
  Average loss when VaR is breached (tail risk)

---

## Project Structure

indian-volatility-risk/
│
├── app.py # Streamlit application
│
├── data/
│ └── universe/
│ └── stocks.yaml # Asset universe (indices & stocks)
│
├── src/
│ └── simulation/
│ ├── monte_carlo.py # GBM simulation logic
│ ├── volatility_sources.py
|
├── src/
│ └── models/
│ ├── ewma.py # GBM simulation logic
│ ├── garch.py
│ ├── ml_models.py 
│ ├── multi_asset_lstm.py
│
├── requirements.txt
└── README.md


---

## Setup Instructions

### 1️. Clone the repository
```bash
git clone https://github.com/<your-username>/indian-volatility-risk.git
cd indian-volatility-risk

2️. Create virtual environment
py -3.11 -m venv venv

3️. Activate environment

Windows

venv\Scripts\activate


Linux / macOS

source venv/bin/activate

4️. Install dependencies
pip install -r requirements.txt

5. Run the Application
streamlit run app.py


The app will open in your browser and always fetch latest available market data.

 Notes & Assumptions

1.Uses Geometric Brownian Motion for price simulation

2.Volatility is assumed constant over forecast horizon

3.Designed for risk analysis, not price prediction

4.Intended for educational & analytical use

 Future Improvements

1.Regime-switching volatility

2.Stochastic volatility models

3.Option pricing & Greeks

4.Scenario-based stress testing

5.Portfolio-level risk aggregation