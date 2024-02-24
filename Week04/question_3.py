import pandas as pd
import numpy as np
from scipy.stats import norm
from function import return_calculate, calculate_VAR

daily_prices_path = 'DailyPrices.csv'
portfolio_path = 'portfolio.csv'
portfolio_df = pd.read_csv(portfolio_path)
daily_prices_df = pd.read_csv(daily_prices_path)

def calculate_portfolio_value(portfolio_df, daily_prices_df):
    daily_prices_long = daily_prices_df.melt(id_vars=["Date"], var_name="Stock", value_name="Price")
    merged_df = pd.merge(portfolio_df, daily_prices_long, on="Stock")
    merged_df["Value"] = merged_df["Holding"] * merged_df["Price"]
    portfolio_values = merged_df.groupby(["Portfolio", "Date"]).sum().reset_index()[["Portfolio", "Date", "Value"]]
    portfolio_values_pivot = portfolio_values.pivot(index="Date", columns="Portfolio", values="Value")
    portfolio_values_pivot['Total Portfolio'] = portfolio_values_pivot.sum(axis=1)

    return portfolio_values_pivot

portfolio_value = calculate_portfolio_value(portfolio_df, daily_prices_df)
print("portfolio value:\n",portfolio_value.head(5))
portfolio_values_reset = portfolio_value.reset_index()
portfolio_return = return_calculate(portfolio_values_reset,method='DISCRETE', dateColumn='Date')
print("portfolio return:\n", portfolio_return.head(5))

#calculate adjusted portfolio return to remove the mean of return
return_mean = portfolio_return.mean(axis=0)
portfolio_return_adjusted = portfolio_return.subtract(return_mean, axis=1)
portfolio_return_adjusted.drop(columns='Date', inplace=True, errors='ignore')

# calculate the percentage VaR
VaR_results_percentage = {}
for column in portfolio_return_adjusted.columns:
    VaR = calculate_VAR(portfolio_return_adjusted[column], alpha=0.05, lambda_ew=0.94, method='Normal with EWV')
    VaR_results_percentage[column] = VaR

print("Percentage VaR for exponential weighted average:\n",VaR_results_percentage)

#Using the Average portfolio value  for dollar VaR
portfolio_last_value = portfolio_value.iloc[-1]
print("portfolio_value_mean:\n", portfolio_last_value)
VaR_results_dollar = {key: VaR_results_percentage[key] * portfolio_last_value[key] for key in VaR_results_percentage}

print("Dollar VaR for exponential weighted average:\n",VaR_results_dollar)

VaR_percentage_normal = {}
for column in portfolio_return_adjusted.columns:
    VaR = calculate_VAR(portfolio_return_adjusted[column], alpha=0.05, lambda_ew=0.94, method='Normal')
    VaR_percentage_normal[column] = VaR
print("Percentage VaR for normal distribution:\n", VaR_percentage_normal)

VaR_dollar_normal = {key: VaR_percentage_normal[key] * portfolio_last_value[key] for key in VaR_percentage_normal}
print("Dollar VaR for normal distribution:\n", VaR_dollar_normal)



"""
lambda_ = 0.94

# Function to calculate VaR of a portfolio
def calculate_portfolio_var(portfolio_return, alpha=0.05):
    portfolio_return_ewm_variance = portfolio_returns.ewm(span=(2 / (1 - lambda_)) - 1).var()
    portfolio_return_ewm_std = np.sqrt(portfolio_return_ewm_variance)
    portfolio_return_mean = portfolio_return.mean()
    VaR = norm.ppf(alpha,portfolio_return_mean, portfolio_return_ewm_std)
    return VaR

# Split portfolio data by portfolio
portfolios = portfolio.groupby('Portfolio')

# Calculate VaR for each portfolio and convert VaR to dollar value
portfolio_vars_dollar = {}
for name, group in portfolios:
    portfolio_weights = group.set_index('Stock').reindex(returns.columns, fill_value=0)['Holding']
    portfolio_values = portfolio_weights * daily_prices[returns.columns].iloc[-1]
    portfolio_vars_dollar[name] = calculate_portfolio_var(ew_cov_matrix, portfolio_values)

# Calculate total VaR (VaR of the total holdings)
total_portfolio = portfolio.groupby('Stock')['Holding'].sum().reindex(returns.columns, fill_value=0)
total_portfolio_values = total_portfolio * daily_prices[returns.columns].iloc[-1]
total_var_dollar = calculate_portfolio_var(ew_cov_matrix, total_portfolio_values)

print("Portfolio Var dollar is:", portfolio_vars_dollar)
print("Total Var dollar is:", total_var_dollar)
"""