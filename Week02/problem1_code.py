import pandas as pd 
import numpy as np
from scipy import stats

def first4Moments(data):
    n = len(data)

    mean = np.sum(data)/n
    variance = np.sum(np.square(data - mean))/(n-1)

    # calculate the biased sigma2 to form the unbiased skewness and kurtosis
    sim_corrected = data - mean
    sigma2 = np.sum(np.square(data - mean))/n

    # normalized skewness = unnormalized skewness/ sigma3
    skewness = n*np.sum((data-mean)**3)/((n-1)*(n-2))/sigma2**1.5
    kurtosis = np.sum(sim_corrected ** 4) / n / (sigma2 ** 2)

    return mean, variance, skewness, kurtosis

file_path = "problem1.csv"
data = pd.read_csv(file_path)
dataset = data['x'].values

m, s2, sk, k = first4Moments(dataset)

print("calculate_Mean:", m)
print("calculate_Variance:", s2)
print("calculate_Skewness:", sk)
print("calculate_Kurtosis:", k)

pd_m = data['x'].mean()
pd_x = data['x'].var()
pd_sk = data['x'].skew()
pd_k = data['x'].kurtosis()

print("pandas_Mean:", pd_m)
print("pandas_Variance:", pd_x)
print("pandas_Skewness:", pd_sk)
print("pandas_Kurtosis:", pd_k)

# question 1c

np.random.seed(0)
theoretical_mean = 50
std_dev = 5
sample_size = 10000
data = np.random.normal(theoretical_mean, std_dev, sample_size)
theoretical_variance = 25
theoretical_skewness = 0
theoretical_kurtosis = 3

series = pd.Series(data)


calculated_mean = series.mean()
calculated_variance = series.var(ddof=1)
calculated_skewness = series.skew()
calculated_kurtosis = series.kurtosis() + 3

t_test_mean = stats.ttest_1samp(series, theoretical_mean)
t_test_variance = stats.ttest_1samp((series**2), theoretical_variance)
t_test_skewness = stats.ttest_1samp(pd.Series([series.skew()] * sample_size), theoretical_skewness)
t_test_kurtosis = stats.ttest_1samp(pd.Series([series.kurtosis()] * sample_size), theoretical_kurtosis)  

# Step 3: Compare to theoretical values
print("\nBelow is the answer for 1c")
print("Calculated Mean:", calculated_mean)
print("Calculated Variance:", calculated_variance)
print("Calculated Skewness:", calculated_skewness)
print("Calculated Kurtosis:", calculated_kurtosis)
print("T-test for Mean:", t_test_mean)
print("T-test for Variance:", t_test_variance)
print("T-test for Skewness:", t_test_skewness)
print("T-test for Kurtosis:", t_test_kurtosis)
