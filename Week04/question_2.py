import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm, t
from function import return_calculate, calculate_VAR


# Load the data
file_path = 'DailyPrices.csv'
prices = pd.read_csv(file_path)

# Calculate arithmetic returns for all assets
arithmetic_returns = return_calculate(prices, method="DISCRETE", dateColumn="Date")
#print(arithmetic_returns.head(5))

# Extracting and adjusting META
meta_return_mean = arithmetic_returns['META'].mean()
meta_adjusted_returns = arithmetic_returns['META'] - meta_return_mean

print(meta_adjusted_returns.head(5))

VaR_normal = calculate_VAR(meta_adjusted_returns, method="Normal")
VaR_normal_ew = calculate_VAR(meta_adjusted_returns, method="Normal with EWV")
VaR_T = calculate_VAR(meta_adjusted_returns, method="T Distribution")
VaR_AR1 = calculate_VAR(meta_adjusted_returns, method="AR1")
VaR_historical = calculate_VAR(meta_adjusted_returns, method="historical")

print("VaR using normal distribution is:", VaR_normal)
print("VaR using normal distribution with exponentially weighted variance is:", VaR_normal_ew )
print("VaR using T distribution is:", VaR_T)
print("VaR using AR1 is:", VaR_AR1)
print("VaR using historical distribution is:", VaR_historical)
