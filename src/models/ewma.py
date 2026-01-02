import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/processed/features.csv")

def ewma_forecast(returns, lambda_=0.94):
    vol = []
    var = returns.var()

    for r in returns:
        var = lambda_ * var + (1 - lambda_) * r**2
        vol.append(np.sqrt(var) * np.sqrt(252))

    return np.array(vol)


def main():
    df = pd.read_csv(DATA_PATH)
    returns = df["log_return"].dropna().values

    ewma_vol = ewma_forecast(returns)
    df.loc[df["log_return"].notna(), "ewma_forecast"] = ewma_vol

    print(df[["date", "vol_20d", "ewma_forecast"]].dropna().head(10))

if __name__ == "__main__":
    main()
