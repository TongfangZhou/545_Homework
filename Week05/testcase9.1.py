from scipy.stats import t, norm, spearmanr
import numpy as np
import pandas as pd

# Assuming test9_1_portfolio.csv and test9_1_returns.csv have been read into
# test9_1_portfolio and test9_1_returns respectively
portfolio_df = pd.read_csv("../testfiles/data/test9_1_portfolio.csv")
returns_df = pd.read_csv("../testfiles/data/test9_1_returns.csv")

# Placeholder function to fit a normal distribution
def fit_normal(data):
    mu, sigma = norm.fit(data)
    return {
        'eval': lambda u: norm.ppf(u, loc=mu, scale=sigma),
        'u': norm.cdf(data, loc=mu, scale=sigma)
    }

# Placeholder function to fit a T distribution
def fit_general_t(data):
    params = t.fit(data)
    nu, mu, sigma = params
    return {
        'eval': lambda u: t.ppf(u, df=nu, loc=mu, scale=sigma),
        'u': t.cdf(data, df=nu, loc=mu, scale=sigma)
    }

# Placeholder function to simulate data using PCA approach
def simulate_pca(cov, nsim, seed=0):
    np.random.seed(seed)
    mean = np.zeros(cov.shape[0])
    data = np.random.multivariate_normal(mean, cov, nsim)
    return data

# Fit models to the returns data
models = {
    "A": fit_normal(returns_df['A']),
    "B": fit_general_t(returns_df['B'])
}

# Spearman correlation for returns data
U = np.column_stack((models["A"]['u'], models["B"]['u']))
corr, _ = spearmanr(U)
cov_matrix = np.array([[1, corr], [corr, 1]])

# Perform PCA-based simulation
nSim = 100000
simulated_data = simulate_pca(cov_matrix, nSim)
simulated_u = norm.cdf(simulated_data)

# Transform simulated quantiles back to returns
simulated_returns = pd.DataFrame({
    "A": models["A"]['eval'](simulated_u[:, 0]),
    "B": models["B"]['eval'](simulated_u[:, 1])
})

# Compute VaR and ES for the portfolio
def compute_risk_metrics(simulated_pnl, alpha=0.05):
    VaR = -np.percentile(simulated_pnl, alpha*100)
    ES = -np.mean(simulated_pnl[simulated_pnl <= -VaR])
    return VaR, ES

# Initialize lists to store results
VaRs, ESs, VaR_pcts, ES_pcts = [], [], [], []

# Compute for each stock and the total portfolio
for stock in portfolio_df['Stock']:
    holding, price = portfolio_df.loc[portfolio_df['Stock'] == stock, ['Holding', 'Starting Price']].values[0]
    pnl = holding * price * simulated_returns[stock]
    VaR, ES = compute_risk_metrics(pnl)
    VaRs.append(VaR)
    ESs.append(ES)
    VaR_pcts.append(VaR / (holding * price))
    ES_pcts.append(ES / (holding * price))

# Add total portfolio metrics
total_pnl = portfolio_df['Holding'].values * portfolio_df['Starting Price'].values @ simulated_returns.T
total_VaR, total_ES = compute_risk_metrics(total_pnl)
VaRs.append(total_VaR)
ESs.append(total_ES)
initial_value = portfolio_df['Holding'].values @ portfolio_df['Starting Price'].values
VaR_pcts.append(total_VaR / initial_value)
ES_pcts.append(total_ES / initial_value)

# Create DataFrame for results
riskOut = pd.DataFrame({
    'Stock': ['A', 'B', 'Total'],
    'VaR95': VaRs,
    'ES95': ESs,
    'VaR95_Pct': VaR_pcts,
    'ES95_Pct': ES_pcts
})

# Display and save the results
riskOut.set_index('Stock', inplace=True)
print(riskOut)

expected = pd.read_csv("../testfiles/data/testout9_1.csv")
print(expected)