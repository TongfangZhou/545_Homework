import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.stats import norm, t
from quant_risk_library import calculate_return_with_date, calculate_VAR
from copulas.multivariate import GaussianMultivariate

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
portfolio_return = calculate_return_with_date(portfolio_values_reset,method='DISCRETE', dateColumn='Date')
print("portfolio return:\n", portfolio_return.head(5))

#calculate adjusted portfolio return to remove the mean of return
numeric_cols = portfolio_return.select_dtypes(include=[np.number])
return_mean = numeric_cols.mean(axis=0)
portfolio_return_adjusted = numeric_cols.subtract(return_mean, axis=1)
portfolio_return_adjusted = portfolio_return_adjusted.dropna().replace([np.inf, -np.inf], np.nan).dropna()



latest_prices = daily_prices_df.iloc[-1, 1:]  # Exclude 'Date' column
# Merge the portfolio holdings with the latest prices
merged_df = pd.merge(portfolio_df, latest_prices.rename('Price'), left_on='Stock', right_index=True)
merged_df['Market Value'] = merged_df['Holding'] * merged_df['Price']
# Group by portfolio and calculate total market value for each portfolio
total_value_by_portfolio = merged_df.groupby('Portfolio')['Market Value'].sum()
# Calculate the weight of each stock in its portfolio
merged_df['Weight'] = merged_df.apply(lambda row: row['Market Value'] / total_value_by_portfolio[row['Portfolio']], axis=1)
# Create the portfolio weight matrix
portfolio_weights_matrix = merged_df.pivot(index='Stock', columns='Portfolio', values='Weight').fillna(0)
portfolio_weights_matrix['Total Portfolio'] = portfolio_weights_matrix.sum(axis=1)
print(portfolio_weights_matrix.head(5))

fitted_a=t.fit(portfolio_return_adjusted["A"])
fitted_b=t.fit(portfolio_return_adjusted["B"])
mean_c, std_c = portfolio_return_adjusted["C"].mean(), portfolio_return_adjusted["C"].std()

copula = GaussianMultivariate()
copula.fit(portfolio_return_adjusted)
num_simulations = 10000
simulated_data = copula.sample(num_simulations)
column_mapping = dict(zip(range(len(portfolio_weights_matrix.index)), portfolio_weights_matrix.index))
aligned_simulated_data = simulated_data.rename(columns=column_mapping)

print("Shape of aligned_simulated_data:", aligned_simulated_data.shape)
print("Shape of portfolio_weights_matrix:", portfolio_weights_matrix.shape)

# Verify the renaming process
if aligned_simulated_data.shape[1] != portfolio_weights_matrix.shape[0]:
    print("The number of columns in aligned_simulated_data does not match the number of stocks in portfolio_weights_matrix.")

# Assuming the shapes are correct, perform the dot product
if aligned_simulated_data.shape[1] == portfolio_weights_matrix.shape[0]:
    portfolio_scenarios = aligned_simulated_data.dot(portfolio_weights_matrix)
else:
    print("Cannot perform dot product due to mismatched dimensions.")

portfolio_scenarios = aligned_simulated_data.dot(portfolio_weights_matrix.T)

var_level = 0.05  # 95% VaR
VaR = np.percentile(portfolio_scenarios, var_level * 100)
ES = portfolio_scenarios[portfolio_scenarios <= VaR].mean()

print(f"VaR: {VaR}, ES: {ES}")
