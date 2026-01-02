import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# -------------------------
# Paths
# -------------------------
DATA_PATH = Path("data/processed/ml_dataset.csv")

# -------------------------
# Utilities
# -------------------------
def time_split(df, ratio=0.8):
    split = int(len(df) * ratio)
    return df.iloc[:split], df.iloc[split:]


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# -------------------------
# Main
# -------------------------
def main():
    df = pd.read_csv(DATA_PATH)

    drop_cols = [
        "date", "target_vol",
        "open", "high", "low", "close", "volume",
        "cum_return", "rolling_max", "drawdown"
    ]

    features = [c for c in df.columns if c not in drop_cols]

    train, test = time_split(df)

    X_train, y_train = train[features], train["target_vol"]
    X_test, y_test = test[features], test["target_vol"]

    print(f"Train samples: {len(train)}")
    print(f"Test samples:  {len(test)}")
    print(f"Features used: {len(features)}")

    # -------- Linear Model --------
    lin = LinearRegression()
    lin.fit(X_train, y_train)

    lin_pred = lin.predict(X_test)
    lin_rmse = rmse(y_test, lin_pred)

    print(f"\nLinear Regression RMSE: {lin_rmse:.6f}")

    # -------- Random Forest --------
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_rmse = rmse(y_test, rf_pred)

    print(f"Random Forest RMSE:   {rf_rmse:.6f}")

    improvement = (lin_rmse - rf_rmse) / lin_rmse * 100
    print(f"\nRF improvement over Linear: {improvement:.2f}%")

    # -------- Feature Importance --------
    importance = pd.Series(
        rf.feature_importances_,
        index=features
    ).sort_values(ascending=False)

    print("\nTop 10 Important Features:")
    print(importance.head(10))


if __name__ == "__main__":
    main()
