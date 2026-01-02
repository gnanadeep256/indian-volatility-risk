import numpy as np

def garch_from_returns(returns):
    from arch import arch_model

    # GARCH expects % returns
    r = returns * 100

    model = arch_model(r, vol="Garch", p=1, q=1)
    res = model.fit(disp="off")

    var = res.forecast(horizon=1).variance.values[-1, 0]
    sigma = np.sqrt(var) / 100

    return float(sigma)


def ml_from_returns(returns, window=20):
    # Recent realized volatility
    return float(returns[-window:].std())


def lstm_from_returns(returns, window=60):
    # DL proxy: longer memory volatility
    return float(returns[-window:].std())

