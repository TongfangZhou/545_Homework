import pandas as pd
import numpy as np
from quant_risk_library import sim_var_es_copula, calculate_portfolio_value, calculate_return_with_date

daily_prices_path = 'DailyPrices.csv'
portfolio_path = 'portfolio.csv'
portfolio_df = pd.read_csv(portfolio_path)
daily_prices_df = pd.read_csv(daily_prices_path)

portfolio_value = calculate_portfolio_value(portfolio_df, daily_prices_df)
portfolio_values_reset = portfolio_value.reset_index()
portfolio_return = calculate_return_with_date(portfolio_values_reset,method='DISCRETE', dateColumn='Date')

#calculate adjusted portfolio return to remove the mean of return
numeric_cols = portfolio_return.select_dtypes(include=[np.number])
return_mean = numeric_cols.mean(axis=0)
portfolio_return_adjusted = numeric_cols.subtract(return_mean, axis=1)
portfolio_return_adjusted = portfolio_return_adjusted.dropna().replace([np.inf, -np.inf], np.nan).dropna()

portfolio_return.to_csv('portfolio_returns_removed.csv', index=False)



# Extract the first row of prices to get the starting prices for each stock
starting_prices = daily_prices_df.iloc[0, 1:]  # Skip the Date column
# Map the starting prices to the corresponding stocks in the portfolio
portfolio_df['Starting Price'] = portfolio_df['Stock'].map(starting_prices)
portfolio_df.to_csv('new_portfolio.csv', index=False)



portfolio_risk = sim_var_es_copula('portfolio_returns_removed.csv', 'new_portfolio.csv', 'DailyPrices.csv')
print(portfolio_risk)
