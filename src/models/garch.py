import pandas as pd
from arch import arch_model
from pathlib import Path
import numpy as np

DATA_PATH = Path("data/processed/features.csv")

def main():
    df = pd.read_csv(DATA_PATH)
    returns = df["log_return"].dropna() * 100  # GARCH prefers %

    model = arch_model(
        returns,
        mean="Zero",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal"
    )

    res = model.fit(disp="off")

    print(res.summary())

    # Forecast next-day volatility
    forecasts = res.forecast(horizon=1)
    garch_vol = np.sqrt(forecasts.variance.values[-1][0]) * np.sqrt(252)

    print("\nNext-day GARCH volatility forecast:", garch_vol)

if __name__ == "__main__":
    main()
