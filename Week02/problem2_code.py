import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import t, norm
from scipy.optimize import minimize


filepath = 'problem2.csv'
data = pd.read_csv(filepath)
X = data['x'].values.reshape(-1,1)
Y = data['y']

# question 2a
# Fitting the OLS model
ols_model = LinearRegression()
ols_model.fit(X, Y)
ols_coefficients = ols_model.coef_
ols_intercept = ols_model.intercept_

# Predictions and residuals for OLS
Y_pred_ols = ols_model.predict(X)
residuals_ols = Y - Y_pred_ols

# Calculate standard deviation of the OLS error
std_dev_ols_error = np.std(residuals_ols)

# calculate standard deviation of MLE
std_dev_mle = np.sqrt(np.sum(residuals_ols ** 2) / len(Y))

print("Below is answer for Q2.a")
print("OLS Coefficients:", ols_coefficients)
print("OLS Intercept:", ols_intercept)
print("Standard Deviation of the OLS Error:", std_dev_ols_error)
print("Fitted MLE σ (Standard Deviation of the Error):", std_dev_mle)

# question 2b
x = data['x']
y = data['y']
def likelihood_t(parameter, y, x):
    intercept, beta, df, sigma = parameter
    prediction = intercept + beta * x
    e = y - prediction
    likelihood_t = np.sum(t.logpdf(e, df=df, scale=sigma))
    return -likelihood_t

initial_parameter_t = [0, 1, 10, 1]
result_t = minimize(likelihood_t, initial_parameter_t, args=(y, x))
likelihood_t = -result_t.fun

intercept_hat_t, beta_hat_t, df_hat, sigma_hat_t = result_t.x

print("T distribution estimated parameters:")
print("Negative Log likelihood of t distribution is:", -likelihood_t)
print("MLE Beta:", beta_hat_t)
print("MLE Intercept:", intercept_hat_t)
print("DF:", df_hat)
print("Sigma:", sigma_hat_t)

# Normal distribution likelihood function
def likelihood_n(parameter, y, x):
    intercept, beta, sigma = parameter
    prediction = intercept + beta * x
    e = y - prediction
    likelihood_n = np.sum(norm.logpdf(e, scale=sigma))
    return -likelihood_n

initial_parameter_n = [np.mean(y), np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1), 1]
result_n = minimize(likelihood_n, initial_parameter_n, args=(y, x), method="L-BFGS-B")
likelihood_n = -result_n.fun

intercept_hat_n, beta_hat_n, sigma_hat_n = result_n.x

print("\nNormal distribution estimated parameters:")
print("Negative Log likelihood of normal distribution is:", -likelihood_n)
print("MLE Beta:", beta_hat_n)
print("MLE Intercept:", intercept_hat_n)
print("Sigma:", sigma_hat_n)

# Compute AIC
num_para_t = 4  # Number of parameters in t-distribution model
aic_t = 2 * num_para_t - 2 * likelihood_t
print("\nAIC for t-distribution models is:", aic_t)

num_para_n = 3  # Number of parameters in normal distribution model
aic_n = 2 * num_para_n - 2 * likelihood_n
print("AIC for normal distribution models is:", aic_n)


# question 2c
file_path_x2 = 'problem2_x.csv'
data_x2 = pd.read_csv(file_path_x2)

# Fitting a multivariate normal distribution to X = [X1, X2]
mean_vector = data_x2.mean()
covariance_matrix = data_x2.cov()

mean_x1 = mean_vector['x1']
mean_x2 = mean_vector['x2']
cov_xx = covariance_matrix.loc['x1', 'x1']
cov_yy = covariance_matrix.loc['x2', 'x2']
cov_xy = covariance_matrix.loc['x1', 'x2']

print("mean_x1, mean_x2, cov_xx, cov_yy, cov_xy: ", mean_x1, mean_x2, cov_xx, cov_yy, cov_xy)


# Function to calculate conditional mean and variance of X2 given X1
def conditional_x2(x1):
    mean_x1 = mean_vector['x1']
    mean_x2 = mean_vector['x2']
    cov_xx = covariance_matrix.loc['x1', 'x1']
    cov_yy = covariance_matrix.loc['x2', 'x2']
    cov_xy = covariance_matrix.loc['x1', 'x2']

    conditional_mean = mean_x2 + cov_xy / cov_xx * (x1 - mean_x1)
    conditional_variance = cov_yy - cov_xy * cov_xy / cov_xx
    return conditional_mean, conditional_variance

x2_conditional_mean, x2_conditional_variance = conditional_x2(data_x2['x1'])
print("x2_conditional_variance:", x2_conditional_variance)

# Calculate conditional mean and 95% confidence interval for each observed X1
observed_x1 = data_x2['x1']
conditional_means = []
confidence_intervals = []
for x1 in observed_x1:
    cond_mean, cond_var = conditional_x2(x1)
    conditional_means.append(cond_mean)
    # 95% confidence interval: mean ± 1.96 * std_dev
    confidence_interval = [cond_mean - 1.96 * np.sqrt(cond_var), cond_mean + 1.96 * np.sqrt(cond_var)]
    confidence_intervals.append(confidence_interval)

# Sorting the observed X1 values for proper plotting
sorted_indices = observed_x1.argsort()
sorted_x1 = observed_x1[sorted_indices]
sorted_conditional_means = np.array(conditional_means)[sorted_indices]
sorted_confidence_intervals = np.array(confidence_intervals)[sorted_indices]

# Plotting with sorted values
plt.figure(figsize=(10, 6))
plt.plot(sorted_x1, sorted_conditional_means, label='Expected Value of X2')
plt.fill_between(sorted_x1, sorted_confidence_intervals[:, 0], sorted_confidence_intervals[:, 1], color='gray', alpha=0.3, label='95% Confidence Interval')
plt.xlabel('Observed X1')
plt.ylabel('Expected X2 with Confidence Interval')
plt.title('Expected Value of X2 given X1 with 95% Confidence Interval')
plt.legend()
plt.show()

# Plotting the observed X1 and X2 in the same graph
plt.figure(figsize=(10, 6))
plt.scatter(data_x2['x1'], data_x2['x2'], color='green', label='Observed X1 and X2')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Scatter Plot of Observed X1 and X2')
plt.legend()
plt.show()
