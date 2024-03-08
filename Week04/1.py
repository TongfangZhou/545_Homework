import pandas as pd
import numpy as np
import math
from scipy.stats import norm, t
from scipy.integrate import quad
from scipy import stats
from scipy.stats import spearmanr
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from numpy.random import default_rng
from scipy.linalg import eigh
from copulas.multivariate import GaussianMultivariate

portfolio = pd.read_csv('portfolio.csv')
daily_prices = pd.read_csv('DailyPrices.csv')

# portfolio value
portfolio_values = pd.DataFrame()
portfolio_values['Date'] = daily_prices['Date']

for p in portfolio['Portfolio'].unique():
    portfolio_values[p] = 0
portfolio_values['Total'] = 0

for index, row in portfolio.iterrows():
    stock = row['Stock']
    holding = row['Holding']
    portfolio_type = row['Portfolio']

    if stock in daily_prices.columns:
        daily_value = daily_prices[stock] * holding
        portfolio_values[portfolio_type] += daily_value
portfolio_values['Total'] = portfolio_values[portfolio['Portfolio'].unique()].sum(axis=1)
print("portfolio_values", portfolio_values.head())


# portfolio returns
portfolio_returns = portfolio_values.copy().iloc[:, 1:] 
portfolio_returns = portfolio_returns.pct_change()
portfolio_returns['Date'] = portfolio_values['Date']
cols = portfolio_returns.columns.tolist()
cols = cols[-1:] + cols[:-1]
portfolio_returns_all = portfolio_returns[cols]
portfolio_returns = portfolio_returns_all.dropna().reset_index(drop=True)
portfolio_returns.head()
print("portfolio_returns", portfolio_returns.head())

portfolio_mean_returns = portfolio_returns.iloc[:, 1:].mean()
portfolio_returns_removed = portfolio_returns.iloc[:, 1:].subtract(portfolio_mean_returns, axis=1)
portfolio_returns_removed['Date'] = portfolio_returns['Date']
cols = portfolio_returns_removed.columns.tolist()
cols = cols[-1:] + cols[:-1]
portfolio_returns_removed = portfolio_returns_removed[cols]
print("portfolio_returns_removed",portfolio_returns_removed.head())

portfolio_returns.to_csv('portfolio_returns_removed.csv', index=False)

# Extract the first row of prices to get the starting prices for each stock
starting_prices = daily_prices.iloc[0, 1:]  # Skip the Date column
# Map the starting prices to the corresponding stocks in the portfolio
portfolio['Starting Price'] = portfolio['Stock'].map(starting_prices)
portfolio.to_csv('new_portfolio.csv', index=False)



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

    random_vars = np.random.randn(nsim,3)

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
    values['simulatedValue'] = values.apply(lambda row: row['currentValue'] * (1 + sim_rtn.loc[row['iteration'] - 1, row['Portfolio']]), axis=1)
    values['pnl'] = values['simulatedValue'] - values['currentValue']
    risk = aggRisk(values, ['Portfolio'])
    
    return risk


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

portfolio_risk = sim_var_es_copula('portfolio_returns_removed.csv', 'new_portfolio.csv', 'DailyPrices.csv')
print(portfolio_risk)
