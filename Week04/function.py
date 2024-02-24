import numpy as np
import pandas as pd
from scipy.stats import norm, t
from statsmodels.tsa.arima.model import ARIMA
def return_calculate(prices, method="DISCRETE", dateColumn="Date"):
    if dateColumn not in prices.columns:
        raise ValueError(f"dateColumn: {dateColumn} not in DataFrame: {list(prices.columns)}")

    # Select columns excluding the date column
    vars = [col for col in prices.columns if col != dateColumn]
    p = prices[vars].values
    n, m = p.shape
    # Initialize an array for calculated returns
    p2 = np.empty((n - 1, m))
    for i in range(n - 1):
        for j in range(m):
            p2[i, j] = p[i + 1, j] / p[i, j]

    # Apply return method
    if method.upper() == "DISCRETE":
        p2 = p2 - 1.0
    elif method.upper() == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError(f"method: {method} must be in (\"LOG\",\"DISCRETE\")")

    # Create output DataFrame
    dates = prices[dateColumn].iloc[1:]
    out = pd.DataFrame({dateColumn: dates})
    for i, var in enumerate(vars):
        out[var] = p2[:, i]

    return out


def calculate_VAR(data, alpha=0.05, lambda_ew=0.94, method="Normal"):
    if method == "Normal":
        mean, std = data.mean(), data.std()
        VaR = norm.ppf(alpha, mean, std)

    elif method == "Normal with EWV":
        """
        ewv_var = data.iloc[:, 1:].ewm(alpha=1-lambda_ew).var().iloc[-1]
        ewv_std_dev = np.sqrt(ewv_var)
        VaR = norm.ppf(0.05) * ewv_std_dev
        """
        variance = data.ewm(alpha=0.06).var()
        last_variance = variance.iloc[-1]
        VaR = norm.ppf(alpha, 0, np.sqrt(last_variance))


    elif method == "T Distribution":
        params = t.fit(data)
        VaR = t.ppf(alpha, *params)

    elif method == "AR1":
        model_arima = ARIMA(data, order=(1, 0, 0))
        model_arima_fit = model_arima.fit()

        forecast = model_arima_fit.get_forecast(steps=1, alpha=0.05)
        VaR = norm.ppf(0.05, loc=forecast.predicted_mean.values[-1], scale=forecast.se_mean.values[-1])

    elif method == "historical":
        VaR = data.quantile(alpha)

    return VaR