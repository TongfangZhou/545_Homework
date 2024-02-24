import numpy as np

p_t0 = 100
sigma = 1
n_simulation = 10000

np.random.seed(0)
r = np.random.normal(0,sigma, n_simulation)

brownian_motion_return = p_t0+r
arithmetic_return = p_t0*(1+r)
log_return = p_t0*np.exp(r)

brownian_mean, brownian_std = np.mean(brownian_motion_return), np.std(brownian_motion_return)
arithmetic_mean, arithmetic_std = np.mean(arithmetic_return), np.std(arithmetic_return)
log_mean, log_std = np.mean(log_return), np.std(log_return)

print("brownian Motion Return Mean:", brownian_mean, "Brownian Motion Return std:", brownian_std)
print("Arithmetic Return Mean:", arithmetic_mean, "Arithmetic Return std:", arithmetic_std)
print("Log Return Mean:", log_mean, "Log Return std:", log_std)