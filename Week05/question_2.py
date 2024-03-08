from quant_risk_library import calculate_VAR, calculate_ES
import pandas as pd
import numpy as np

file_path = "problem1.csv"
x1=pd.read_csv(file_path)



# getting VaR from calculate_VAR function
VaR_ewv = calculate_VAR(x1['x'], alpha=0.05, lambda_ew=0.94, method = "Normal with EWV")
VaR_T = calculate_VAR(x1['x'], alpha=0.05, lambda_ew=0.94, method = "T Distribution")
VaR_historic = calculate_VAR(x1['x'], alpha=0.05, lambda_ew=0.94, method = "historical")

print("VaR for normal distribution with exponential weighted average is:", VaR_ewv)
print("VaR for T distribution is:", VaR_T)
print("VaR for historical simulation is:", VaR_historic)

# calculating the expected shortfall
ES_ewv = calculate_ES(x1['x'], alpha=0.05, lambda_ew=0.94, method = "Normal with EWV")
ES_T = calculate_ES(x1['x'], alpha=0.05, lambda_ew=0.94, method = "T Distribution")
ES_historic = calculate_ES(x1['x'], alpha=0.05, lambda_ew=0.94, method = "historical")

print("ES for normal distribution with exponential weighted average is:", ES_ewv)
print("ES for T distribution is:", ES_T)
print("ES for historical simulation is:", ES_historic)