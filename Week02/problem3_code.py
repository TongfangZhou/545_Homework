import pandas as pd
import statsmodels.tsa.api as smt
import itertools
file_path = 'problem3.csv'  # Replace with your file path
data = pd.read_csv(file_path)
series = data.iloc[:, 0]

d = 0

q = 0
best_aic_AR = float("inf")
best_order_AR = None
best_model_AR = None

for p in range(1,4):
    AR_model = smt.ARIMA(series, order=(p, d, q))
    AR_results = AR_model.fit()
    if AR_results.aic < best_aic_AR:
        best_aic_AR = AR_results.aic
        best_order_AR = p
        best_model_AR = AR_results

print("The best AR model's order and aic is:", best_order_AR, best_aic_AR)

p = 0
best_aic_MA = float("inf")
best_order_MA = None
best_model_MA = None

for q in range(1, 4):
    MA_model = smt.ARIMA(series, order=(p, d, q))
    MA_results = MA_model.fit()
    if MA_results.aic < best_aic_MA:
        best_aic_MA = MA_results.aic
        best_order_MA = q
        best_model_MA = MA_results

print("The best MA model's order and aic is:", best_order_MA, best_aic_MA)

if best_aic_MA < best_aic_AR:
    print("MA is a better solution")
    print("Best AIC: ", best_aic_MA)
    print("Best Order: ", best_order_MA)
    print(best_model_MA.summary())
else:
    print("AR is a better solution")
    print("Best AIC: ", best_aic_AR)
    print("Best Order: ", best_order_AR)
    print(best_model_AR.summary())

