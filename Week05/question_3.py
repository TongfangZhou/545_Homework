import pandas as pd
from scipy.stats import t, norm
from quant_risk_library import calculate_return_with_date, fit_distribution
import numpy as np
from scipy.stats import multivariate_normal

daily_prices = pd.read_csv("DailyPrices.csv")
portfolio = pd.read_csv("portfolio.csv")

# Calculate returns and adjust
daily_return = calculate_return_with_date(daily_prices, "DISCRETE", "Date")
daily_return_mean = daily_return.mean(numeric_only=True)  # Fix for FutureWarning
daily_return_adjusted = daily_return.subtract(daily_return_mean, axis=1)
daily_return_adjusted.drop(columns='Date', inplace=True, errors='ignore')

# Fit distributions
portfolio_a = portfolio[portfolio['Portfolio'] == 'A']
portfolio_b = portfolio[portfolio['Portfolio'] == 'B']
portfolio_c = portfolio[portfolio['Portfolio'] == 'C']

fitted_a = fit_distribution(daily_return_adjusted, portfolio_a, 't')
fitted_b = fit_distribution(daily_return_adjusted, portfolio_b, 't')
fitted_c = fit_distribution(daily_return_adjusted, portfolio_c, 'normal')

def create_gaussian_copula(returns, portfolio):
    portfolio_returns = returns[portfolio['Stock'].unique()]
    correlation_matrix = portfolio_returns.corr(method='spearman').values

    # Generate samples from a multivariate normal distribution
    mvn = multivariate_normal(mean=np.zeros(len(portfolio_returns.columns)), cov=correlation_matrix)
    samples = mvn.rvs(size=10000)

    # Transform samples to uniform using the CDF of a standard normal
    uniform_samples = norm.cdf(samples)
    return uniform_samples


# Updated simulate_portfolio_returns function for improved performance
def simulate_portfolio_returns(uniform_samples, fitted_params, portfolio):
    simulated_returns_list = []

    for i, stock in enumerate(portfolio['Stock'].unique()):
        if len(fitted_params[stock]) == 3:  # t-distribution
            stock_returns = t.ppf(uniform_samples[:, i], *fitted_params[stock])
        elif len(fitted_params[stock]) == 2:  # normal distribution
            stock_returns = norm.ppf(uniform_samples[:, i], *fitted_params[stock])
        simulated_returns_list.append(pd.Series(stock_returns, name=stock))

    simulated_returns = pd.concat(simulated_returns_list, axis=1)
    portfolio_returns = (simulated_returns * portfolio.set_index('Stock')['Holding']).sum(axis=1)
    return portfolio_returns

def calculate_portfolio_risk(portfolio_returns, alpha=0.05):
    # Calculate VaR and ES
    var = np.percentile(portfolio_returns, alpha * 100)
    es = portfolio_returns[portfolio_returns <= var].mean()
    return var, es


# Create Gaussian copulas for each portfolio
uniform_samples_a = create_gaussian_copula(daily_return_adjusted, portfolio_a)
uniform_samples_b = create_gaussian_copula(daily_return_adjusted, portfolio_b)
uniform_samples_c = create_gaussian_copula(daily_return_adjusted, portfolio_c)

# Simulate returns
simulated_returns_a = simulate_portfolio_returns(uniform_samples_a, fitted_a, portfolio_a)
simulated_returns_b = simulate_portfolio_returns(uniform_samples_b, fitted_b, portfolio_b)
simulated_returns_c = simulate_portfolio_returns(uniform_samples_c, fitted_c, portfolio_c)

# Calculate risks
var_a, es_a = calculate_portfolio_risk(simulated_returns_a)
var_b, es_b = calculate_portfolio_risk(simulated_returns_b)
var_c, es_c = calculate_portfolio_risk(simulated_returns_c)

# Aggregate risks
total_var = var_a + var_b + var_c
total_es = es_a + es_b + es_c

print("VaR A:", var_a)
print("VaR B:", var_b)
print("VaR C:", var_c)
print("ES A:", es_a)
print("ES B:", es_b)
print("ES C:", es_c)
