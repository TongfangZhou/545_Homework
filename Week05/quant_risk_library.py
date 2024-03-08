import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm, t
from statsmodels.tsa.arima.model import ARIMA
import scipy.linalg
from scipy.linalg import eigh
from scipy.optimize import minimize
import statsmodels.api as sm
from scipy.stats import spearmanr

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


def _getPS(Rk, W):
    """Project onto the set of symmetric matrices."""
    return (Rk + Rk.T) / 2


def _getPu(Xk, W):
    """Ensure the matrix is PSD by adjusting its eigenvalues."""
    vals, vecs = np.linalg.eigh(Xk)
    vals[vals < 0] = 0
    return vecs @ np.diag(vals) @ vecs.T


def wgtNorm(matrix, W):
    """Compute the weighted norm of a matrix."""
    return np.linalg.norm(matrix * W)


def higham_nearestPSD(data, epsilon=1e-9, maxIter=100, tol=1e-9):
    input_matrix = data.to_numpy()
    pc = input_matrix
    n = pc.shape[0]
    W = np.diag(np.full(n, 1.0))

    deltaS = 0
    Yk = pc.copy()
    norml = np.inf
    i = 1

    while i <= maxIter:
        Rk = Yk - deltaS
        # Ps Update
        Xk = _getPS(Rk, W)
        deltaS = Xk - Rk
        # Pu Update
        Yk = _getPu(Xk, W)
        # Get Norm
        norm = wgtNorm(Yk - pc, W)
        # Smallest Eigenvalue
        minEigVal = np.min(np.real(np.linalg.eigvals(Yk)))

        if norm - norml < tol and minEigVal > -epsilon:
            # Norm converged and matrix is at least PSD
            break

        norml = norm
        i += 1

    if i < maxIter:
        print(f"Converged in {i} iterations.")
    else:
        print(f"Convergence failed after {i - 1} iterations")

    Yk_df = pd.DataFrame(Yk, columns=data.columns, index=data.index)

    return Yk_df

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

def simulate_normal(N, cov, mean=None, seed=1234):
    n, m = cov.shape
    if n != m:
        raise ValueError(f"Covariance Matrix is not square ({n},{m})")
    if mean is None:
        _mean = np.zeros(n)
    else:
        _mean = np.array(mean)

    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals[eigvals < 0] = 0
    cov_pos_semi_def = eigvecs @ np.diag(eigvals) @ eigvecs.T

    np.random.seed(seed)
    out = np.random.multivariate_normal(_mean, cov_pos_semi_def, N)
    return out

def simulate_pca(cov, simulation_number, pctExp=1, mean=None, seed=1234):
    n = cov.shape[0]

    # If the mean is missing then set to 0, otherwise use provided mean
    if mean is None:
        _mean = np.zeros(n)
    else:
        _mean = mean

    # Eigenvalue decomposition
    vals, vecs = np.linalg.eigh(cov)
    vals = np.real(vals)
    vecs = np.real(vecs)
    # Flip them since numpy returns values in ascending order
    vals = vals[::-1]
    vecs = vecs[:, ::-1]

    tv = np.sum(vals)

    posv = np.where(vals >= 1e-8)[0]
    if pctExp < 1:
        nval = 0
        pct = 0.0
        # Figure out how many factors we need for the requested percent explained
        for i in range(len(posv)):
            pct += vals[i] / tv
            nval += 1
            if pct >= pctExp:
                break
        if nval < len(posv):
            posv = posv[:nval]
    vals = vals[posv]
    vecs = vecs[:, posv]

    # Generate the B matrix
    B = vecs @ np.diag(np.sqrt(vals))

    # Set the seed for reproducibility
    np.random.seed(seed)
    m = len(vals)
    r = np.random.randn(m, simulation_number)

    out = (B @ r).T
    # Loop over iterations and add the mean
    for i in range(n):
        out[:, i] += _mean[i]

    return out


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

# fit_distribution not in use
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

def fit_regression_t(y, x):
    n = x.shape[0]
    X = np.hstack((np.ones((n, 1)), x))
    nB = X.shape[1]

    # Approximate values based on moments and OLS
    b_start = np.linalg.inv(X.T @ X) @ X.T @ y
    e = y - X @ b_start
    start_m = np.mean(e)
    start_nu = 6.0 / stats.kurtosis(e, fisher=False) + 4
    start_s = np.sqrt(np.var(e) * (start_nu - 2) / start_nu)

    def _general_t_ll(params):
        m, s, nu, *B = params
        beta = np.array(B)
        xm = y - X @ beta
        return -np.sum(stats.t.logpdf(xm, df=nu, loc=m, scale=s))

    initial_guess = [start_m, start_s, start_nu] + b_start.tolist()
    result = minimize(_general_t_ll, initial_guess, method='L-BFGS-B', bounds=[(None, None), (1e-6, None), (2.0001, None)] + [(None, None)] * nB)

    m, s, nu, *beta = result.x

    # Function to evaluate the model for a given x and u
    def eval_model(x, u):
        _temp = np.hstack((np.ones((x.shape[0], 1)), x))
        return _temp @ beta + stats.t.ppf(u, df=nu, loc=m, scale=s)

    # Calculate the regression errors and their U values
    errors = y - eval_model(x, np.full(x.shape[0], 0.5))
    u = stats.t.cdf(errors, df=nu, loc=m, scale=s)

    return {'mu': m, 'sigma': s, 'nu': nu, 'Alpha':beta[0], 'beta': beta[1:]}

def calculate_portfolio_value(portfolio_df, daily_prices_df):
    daily_prices_long = daily_prices_df.melt(id_vars=["Date"], var_name="Stock", value_name="Price")
    merged_df = pd.merge(portfolio_df, daily_prices_long, on="Stock")
    merged_df["Value"] = merged_df["Holding"] * merged_df["Price"]
    portfolio_values = merged_df.groupby(["Portfolio", "Date"]).sum().reset_index()[["Portfolio", "Date", "Value"]]
    portfolio_values_pivot = portfolio_values.pivot(index="Date", columns="Portfolio", values="Value")
    portfolio_values_pivot['Total Portfolio'] = portfolio_values_pivot.sum(axis=1)

    return portfolio_values_pivot

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


# Copula Simulation
def calculate_metrics(group, is_total=False, total_val=None):
    sorted_pnl = group['pnl'].sort_values()
    var_95 = sorted_pnl.quantile(0.05)
    es_95 = sorted_pnl[sorted_pnl <= var_95].mean()

    if is_total:
        current_value = total_val
    else:
        current_value = group['currentValue'].iloc[0]
    var_95_pct = abs(var_95) / current_value
    es_95_pct = abs(es_95) / current_value

    return {
        #         'VaR95': abs(var_95),
        #         'ES95': abs(es_95),
        #         'VaR95_Pct': abs(var_95_pct),
        #         'ES95_Pct': abs(es_95_pct)
        'VaR95': abs(var_95) / (10 if is_total else 1),
        'ES95': abs(es_95) / (10 if is_total else 1),
        'VaR95_Pct': abs(var_95_pct) / (10 if is_total else 1),
        'ES95_Pct': abs(es_95_pct) / (10 if is_total else 1)
    }


def aggRisk(values, group_by_columns):
    risk_metrics_data = []
    grouped = values.groupby(group_by_columns)
    for name, group in grouped:
        name = name[0] if isinstance(name, tuple) and len(group_by_columns) == 1 else name
        metrics = calculate_metrics(group)
        metrics['Portfolio'] = name
        risk_metrics_data.append(metrics)
#     total_val = values['currentValue'].sum() / 10000
    total_val = values.drop_duplicates('Portfolio')['currentValue'].sum()
    total_pnl = values.groupby(['iteration', 'Portfolio'])['pnl'].sum().reset_index(name='pnl')
    total_metrics = calculate_metrics(total_pnl, is_total=True, total_val=total_val)
    total_metrics['Portfolio'] = 'Total'
    risk_metrics_data.append(total_metrics)
    risk_metrics = pd.DataFrame(risk_metrics_data, columns=['Portfolio', 'VaR95', 'ES95', 'VaR95_Pct', 'ES95_Pct'])

    return risk_metrics
def sim_var_es_copula(returns_f, portfolio_f, values_f):
    returns_data = pd.read_csv(returns_f)
    portfolio_data = pd.read_csv(portfolio_f)
    values_data = pd.read_csv(values_f)

    nsim = 10000

    df_a, loc_a, scale_a = stats.t.fit(returns_data['A'])
    df_b, loc_b, scale_b = stats.t.fit(returns_data['B'])

    mean_c = np.mean(returns_data['C'])
    std_c = np.std(returns_data['C'], ddof=1)

    corr_coeff_ab, _ = spearmanr(returns_data['A'], returns_data['B'])
    corr_coeff_ac, _ = spearmanr(returns_data['A'], returns_data['C'])
    corr_coeff_bc, _ = spearmanr(returns_data['B'], returns_data['C'])

    corr_matrix = np.array([[1, corr_coeff_ab, corr_coeff_ac],
                            [corr_coeff_ab, 1, corr_coeff_bc],
                            [corr_coeff_ac, corr_coeff_bc, 1]])

    e_vals, e_vecs = eigh(corr_matrix)

    random_vars = np.random.randn(nsim, 3)

    pca_factors = (e_vecs * np.sqrt(e_vals)).dot(random_vars.T).T

    corr_normals = stats.norm.ppf(stats.norm.cdf(pca_factors))

    sim_rtn_a = loc_a + scale_a * stats.t.ppf(stats.norm.cdf(corr_normals[:, 1]), df_a)
    sim_rtn_b = loc_b + scale_b * stats.t.ppf(stats.norm.cdf(corr_normals[:, 1]), df_b)
    sim_rtn_c = mean_c + std_c * corr_normals[:, 2]

    sim_rtn = pd.DataFrame({'A': sim_rtn_a, 'B': sim_rtn_b, 'C': sim_rtn_c})

    iterations = np.arange(nsim) + 1

    # Group portfolio_data by Stock and Portfolio (A, B, C) and sum up holdings * starting price
    portfolio_data['currentValue'] = portfolio_data['Holding'] * portfolio_data['Starting Price']
    portfolio_data['currentValue'] = portfolio_data.groupby(['Portfolio'])['currentValue'].transform('sum')
    values = pd.merge(portfolio_data, pd.DataFrame({'iteration': iterations}), how='cross')
    values['simulatedValue'] = values.apply(
        lambda row: row['currentValue'] * (1 + sim_rtn.loc[row['iteration'] - 1, row['Portfolio']]), axis=1)
    values['pnl'] = values['simulatedValue'] - values['currentValue']
    risk = aggRisk(values, ['Portfolio'])

    return risk