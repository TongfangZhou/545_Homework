import numpy as np
import pandas as pd
from scipy.stats import norm, t
from statsmodels.tsa.arima.model import ARIMA
from scipy.linalg import eigh
from scipy.optimize import minimize
import statsmodels.api as sm

def calculate_covariance(data, method="skip missing rows"):
    if method.lower() == "skip missing rows":
        valid_data = data.dropna()
        cov_matrix = valid_data.cov()
    elif method.lower() == "pairwise":
        cov_matrix = data.cov(min_periods=1)
    else:
        raise ValueError("Invalid method. Choose 'skip missing rows' or 'pairwise'.")

    return cov_matrix

def calculate_correlation(data, method="skip missing rows"):
    if method.lower() == "skip missing rows":
        # Skip rows with any missing values
        valid_data = data.dropna()
        corr_matrix = valid_data.corr()
    elif method.lower() == "pairwise":
        # Calculate correlation using pairwise deletion of missing data
        corr_matrix = data.corr(min_periods=1)
    else:
        raise ValueError("Invalid method. Choose 'skip missing rows' or 'pairwise'.")

    return corr_matrix

def exponentially_weighted_covariance(data, lambda_):
    # Calculate the weights
    n = len(data)
    weights = np.array([(1 - lambda_) * (lambda_ ** (n - 1 - i)) for i in range(n)])
    weights /= weights.sum()
    # De-mean the data
    demeaned_data = data - data.mean()

    ew_cov_matrix = (demeaned_data.T * weights).dot(demeaned_data) / weights.sum()
    return ew_cov_matrix

def exponentially_weighted_correlation(data, lambda_):
    ew_cov_matrix = exponentially_weighted_covariance(data, lambda_)
    std_dev = np.sqrt(np.diag(ew_cov_matrix))
    ew_corr_matrix = ew_cov_matrix / np.outer(std_dev, std_dev)
    return ew_corr_matrix


def near_psd(a, epsilon=0.0):
    n = a.shape[0]

    # Copy the matrix as 'out'
    out = np.array(a, copy=True)

    # Calculate the correlation matrix if we got a covariance
    if not np.allclose(np.diag(out), 1):
        invSD = np.diag(1.0 / np.sqrt(np.diag(out)))
        out = np.dot(invSD, np.dot(out, invSD))

    # Eigenvalue decomposition
    vals, vecs = eigh(out)
    # Clamp negative eigenvalues to epsilon
    vals = np.maximum(vals, epsilon)

    # Recalculate the matrix
    T = np.diag(1.0 / np.sqrt(np.sum(vecs ** 2 * vals, axis=1)))
    l = np.diag(np.sqrt(vals))
    B = np.dot(np.dot(T, vecs), l)
    out = np.dot(B, B.T)

    # Add back the variance if necessary
    if 'invSD' in locals():
        invSD = np.diag(1.0 / np.diag(invSD))
        out = np.dot(invSD, np.dot(out, invSD))

    return out


def chol_psd(a):
    if isinstance(a, pd.DataFrame):
        a = a.to_numpy()

    n = a.shape[0]
    root = np.zeros_like(a)

    for j in range(n):
        s = 0.0
        if j > 0:
            s = np.dot(root[j, :j], root[j, :j])

        temp = a[j, j] - s
        if temp < 0 and np.abs(temp) < 1e-8:
            temp = 0.0

        root[j, j] = np.sqrt(temp)

        if root[j, j] != 0:
            ir = 1.0 / root[j, j]
            for i in range(j + 1, n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i, j] = (a[i, j] - s) * ir

    np.round(root, 6)
    return root

def calculate_return_with_date(prices, method="DISCRETE", dateColumn="Date"):
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

def calculate_return_without_date(prices, method="DISCRETE"):
    # No need to check for dateColumn now
    p = prices.values
    n, m = p.shape
    # Initialize an array for calculated returns
    p2 = np.empty((n - 1, m))
    for i in range(n - 1):
        for j in range(m):
            if method.upper() == "DISCRETE":
                p2[i, j] = p[i + 1, j] / p[i, j] - 1
            elif method.upper() == "LOG":
                p2[i, j] = np.log(p[i + 1, j] / p[i, j])
            else:
                raise ValueError(f"method: {method} must be in (\"LOG\",\"DISCRETE\")")

    # Create output DataFrame without the 'Date' column
    return pd.DataFrame(p2, columns=prices.columns)

def fit_normal_distribution(data):
    mu, std = data.mean(), data.std()
    return pd.DataFrame({'mu': mu, 'sigma': std})

def fit_t_distribution(data):
    df, loc, scale = t.fit(data)
    return pd.DataFrame({'mu': [loc],'sigma': [scale], 'nu': [df]})

def fit_distribution(returns, portfolio, distribution='t'):
    """Fit a distribution to the returns of stocks in the portfolio."""
    fitted_params = {}
    for stock in portfolio['Stock'].unique():
        if stock in returns.columns:
            if distribution == 't':
                params = t.fit(returns[stock].dropna())
            elif distribution == 'normal':
                params = (returns[stock].mean(), returns[stock].std())
            fitted_params[stock] = params
    return fitted_params

def t_regression(data):
    X = data.iloc[:, :-1]  # independent variables
    y = data.iloc[:, -1]   # dependent variable

    # Add a column of ones for the intercept
    X = sm.add_constant(X)

    # Negative log likelihood function
    def neg_log_likelihood(params):
        alpha, beta, sigma, nu = params[0], params[1:-2], params[-2], params[-1]
        y_pred = alpha + np.dot(X, beta)
        residuals = y - y_pred
        return -np.sum(t.logpdf(residuals, df=nu, loc=0, scale=sigma))

    # Initial guesses
    params_init = np.concatenate([[0], np.zeros(X.shape[1]), [1, 10]])  # alpha, betas, sigma, nu

    # Bounds to ensure sigma and nu are positive
    bounds = [(None, None)] * (X.shape[1] + 1) + [(1e-5, None), (1e-5, None)]

    # Optimize the negative log likelihood function
    result = minimize(neg_log_likelihood, x0=params_init, bounds=bounds)

    if result.success:
        alpha, betas, sigma, nu = result.x[0], result.x[1:-2], result.x[-2], result.x[-1]
        return {
            "mu": alpha,
            "sigma": sigma,
            "nu": nu,
            "Alpha": alpha,
            "B1": betas[0],
            "B2": betas[1],
            "B3": betas[2]
        }
    else:
        raise ValueError("Optimization failed")

def calculate_VAR(data, alpha=0.05, lambda_ew=0.94, method="Normal"):
    if method == "Normal":
        mean, std = data.mean(), data.std()
        VaR = norm.ppf(alpha, mean, std)

    elif method == "Normal with EWV":
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

    return -VaR


def calculate_ES(data, alpha=0.05, lambda_ew=0.94, method="Normal"):
    VaR = calculate_VAR(data, alpha, lambda_ew, method)
    neg_VaR = -VaR

    if method == "Normal":
        mean, std = data.mean(), data.std()
        ES = -mean + norm.pdf(norm.ppf(alpha)) / alpha * std

    elif method == "Normal with EWV":
        mean = np.mean(data)
        weights = np.array([(lambda_ew ** i) for i in range(len(data))])
        weights /= weights.sum()
        squared_diffs = (data - mean) ** 2
        ewma_variance = np.sum(weights * squared_diffs[::-1])
        ewma_std = np.sqrt(ewma_variance)
        ES = -mean + ewma_std * norm.pdf(norm.ppf(alpha)) / alpha

    elif method == "T Distribution":
        nu, mu, sigma = t.fit(data)
        VaR = t.ppf(alpha, nu, loc=mu, scale=sigma)
        ES = -(mu - (sigma * (t.pdf(t.ppf(alpha, nu), nu) / alpha) * ((nu + (VaR - mu) ** 2 / sigma ** 2) / (nu - 1))))

    elif method == "historical":
        sorted_data = np.sort(data)
        cutoff_index = int(np.floor(alpha * len(sorted_data)))
        ES = -(np.mean(sorted_data[:cutoff_index]))

    return ES