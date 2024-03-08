from quant_risk_library import calculate_covariance, calculate_correlation, exponentially_weighted_covariance, exponentially_weighted_correlation
from quant_risk_library import near_psd, chol_psd, calculate_return_with_date, fit_normal_distribution, fit_t_distribution, fit_regression_t
from quant_risk_library import calculate_VAR, calculate_ES
from quant_risk_library import higham_nearestPSD
from quant_risk_library import simulate_normal, simulate_pca
import pandas as pd

# Function to check if two files are equal with small tolerance set
# test and see if results are the same
def testfiles_equal(df1, df2):
    if df1.equals(df2):
        return True
    else:
        # If they are not equal, it could be due to floating point precision,
        # so we attempt a comparison that allows for a small absolute difference.
        try:
            pd.testing.assert_frame_equal(df1, df2, atol=1e-3)  # Use an appropriate tolerance
            return True
        except AssertionError:
            return False

data_path_1 = "../testfiles/data/test1.csv"
data_1 = pd.read_csv(data_path_1)
cov1_calculated = calculate_covariance(data_1, method="pairwise")
cov1_expected = pd.read_csv("../testfiles/data/testout_1.3.csv")
#print("Calculated Cov when pairwise:\n", cov1_calculated)
#print("Expected Cov when pairwise:\n", cov1_expected)
#print("Test 1.1 calculated Value and Expected value is indifferent with 1e-3 tolerence:", testfiles_equal(cov1_calculated,cov1_expected))

corr1_calculated = calculate_correlation(data_1,method="pairwise")
corr1_expected = pd.read_csv("../testfiles/data/testout_1.4.csv")
#print("Calculated Correlation when pairwise:\n", corr1_calculated)
#print("Expected Correlation when pairwise:\n", corr1_expected)

data_path_2 = "../testfiles/data/test2.csv"
data_2 = pd.read_csv(data_path_2)
ewcov_calculated = exponentially_weighted_covariance(data_2,0.97)
ewcov_expected = pd.read_csv("../testfiles/data/testout_2.1.csv")
#print("Calculated exponentially weighted covariance is:\n", ewcov_calculated)
#print("Expected exponentially weighted covariance is:\n", ewcov_expected)

ewcorr_calculated = exponentially_weighted_correlation(data_2, 0.94)
ewcorr_expected = pd.read_csv("../testfiles/data/testout_2.2.csv")
#print("Calculated exponentially weighted correlation is:\n", ewcorr_calculated)
#print("Expected exponentially weighted correlation is:\n", ewcorr_expected)

NPD_calculated = near_psd(cov1_expected)
NPD_expected = pd.read_csv("../testfiles/data/testout_3.1.csv")
#print("Calculated nearest positive definite matrix for covariance matrix is:\n", NPD_calculated)
#print("Expected nearest positive definite matrix for covariance matrix is:\n", NPD_expected)

higham_input = pd.read_csv("../testfiles/data/testout_1.3.csv")
higham_cov_calculated = higham_nearestPSD(higham_input, epsilon=1e-9, maxIter=100, tol=1e-9)
higham_corr_expected = pd.read_csv("../testfiles/data/testout_3.3.csv")
#print("Calculated nearest positive definite matrix for covariance matrix is:\n", higham_cov_calculated)
#print("Expected nearest positive definite matrix for covariance matrix is:\n", higham_corr_expected)

higham_corr_calculated = higham_nearestPSD(corr1_expected, epsilon=1e-9, maxIter=100, tol=1e-9)
higham_corr_expected = pd.read_csv("../testfiles/data/testout_3.4.csv")
#print("Calculated nearest positive definite matrix for correlation matrix is:\n", higham_corr_calculated)
#print("Expected nearest positive definite matrix for correlation matrix is:\n", higham_corr_expected)

chol_calculated = chol_psd(NPD_expected)
chol_expected = pd.read_csv("../testfiles/data/testout_4.1.csv")
#print("Calculated chol PDS is:\n", chol_calculated)
#print("Expected chol PDS is:\n", chol_expected)

# 5.1 Normal PD simulation
simultion_5_1=pd.read_csv("../testfiles/data/test5_1.csv")
normal_pd_simulation = simulate_normal(100000, simultion_5_1, mean=None, seed=1234)
normal_pd_simulation = pd.DataFrame(normal_pd_simulation)
normal_pd_cov = normal_pd_simulation.cov()
normal_pd_expected = pd.read_csv("../testfiles/data/testout_5.1.csv")
#print(normal_pd_cov)
#print(normal_pd_expected)

# 5.2 Normal PSD simulation
simultion_5_2=pd.read_csv("../testfiles/data/test5_2.csv")
normal_psd_simulation = simulate_normal(100000, simultion_5_2, mean=None, seed=1234)
normal_psd_simulation = pd.DataFrame(normal_pd_simulation)
normal_psd_cov = normal_psd_simulation.cov()
normal_psd_expected = pd.read_csv("../testfiles/data/testout_5.2.csv")
#print(normal_psd_cov)
#print(normal_psd_expected)

# 5.3 near_PDS fixed Normal simulation
simultion_5_3=pd.read_csv("../testfiles/data/test5_3.csv")
fixed_simultion_5_3=near_psd(simultion_5_3)
normal_near_psd_simulation = simulate_normal(100000, fixed_simultion_5_3, mean=None, seed=1234)
normal_near_psd_simulation = pd.DataFrame(normal_near_psd_simulation)
normal_near_psd_cov = normal_near_psd_simulation.cov()
normal_psd_expected = pd.read_csv("../testfiles/data/testout_5.3.csv")
#print(normal_near_psd_cov)
#print(normal_psd_expected)

# 5.4 higham fix Normal simulation
fixed_simulation_5_3=higham_nearestPSD(simultion_5_3, epsilon=1e-9, maxIter=100, tol=1e-9)
normal_higham_simulation = simulate_normal(100000, fixed_simulation_5_3, mean=None, seed=1234)
normal_higham_simulation = pd.DataFrame(normal_higham_simulation)
normal_higham_cov = normal_higham_simulation.cov()
normal_higham_expected = pd.read_csv("../testfiles/data/testout_5.4.csv")
#print(normal_higham_cov)
#print(normal_higham_expected)


# 5.5 PCA simulation
simultion_5_2=pd.read_csv("../testfiles/data/test5_2.csv")
pca_simulation = simulate_pca(simultion_5_2,100000, pctExp=0.99, mean=None, seed=1234)
pca_simulation = pd.DataFrame(pca_simulation)
pca_cov = pca_simulation.cov()
pca_expected = pd.read_csv("../testfiles/data/testout_5.5.csv")
#print(pca_cov)
#print(pca_expected)

# 6.1
data_path_6 = "../testfiles/data/test6.csv"
data_6 = pd.read_csv(data_path_6)
arith_return_calculated = calculate_return_with_date(data_6, method="DISCRETE", dateColumn="Date")
arith_return_expected = pd.read_csv("../testfiles/data/test6_1.csv")
#print(arith_return_calculated.head(5))
#print(arith_return_expected.head(5))

# 6.2
log_return_calculated = calculate_return_with_date(data_6, method="LOG", dateColumn="Date")
log_return_expected = pd.read_csv("../testfiles/data/test6_2.csv")

# 7.1
data_7_1= pd.read_csv("../testfiles/data/test7_1.csv")
normal_para_calculated = fit_normal_distribution(data_7_1)
normal_para_expected = pd.read_csv("../testfiles/data/testout7_1.csv")
#print(normal_para_calculated.head(5))
#print(normal_para_expected.head(5))

# 7.2
data_7_2 = pd.read_csv("../testfiles/data/test7_2.csv")
t_para_calculated = fit_t_distribution(data_7_2)
t_para_expected = pd.read_csv("../testfiles/data/testout7_2.csv")
#print(t_para_calculated.head(5))
#print(t_para_expected.head(5))

# 7.3
data_7_3 = pd.read_csv("../testfiles/data/test7_3.csv")
x_values = data_7_3[['x1', 'x2', 'x3']].values
y_values = data_7_3['y'].values
t_regression_calculated = fit_regression_t(y_values, x_values)
t_regression_expected = pd.read_csv("../testfiles/data/testout7_3.csv")
print(t_regression_calculated)
print(t_regression_expected)

# 8.1
VAR_8_1_calculated = calculate_VAR(data_7_1,0.05,0.94,method="Normal")
VAR_8_1_expected = pd.read_csv("../testfiles/data/testout8_1.csv")
#print(VAR_8_1_calculated)
#print(VAR_8_1_expected)

# 8.2
VAR_8_2_calculated = calculate_VAR(data_7_2,0.05,0.94,method="T Distribution")
VAR_8_2_expected = pd.read_csv("../testfiles/data/testout8_2.csv")
#print(VAR_8_2_calculated)
#print(VAR_8_2_expected)

# 8.3
VAR_8_3_calculated = calculate_VAR(data_7_2,0.05,0.94,method="historical")
VAR_8_3_expected = pd.read_csv("../testfiles/data/testout8_3.csv")
#print(VAR_8_3_calculated)
#print(VAR_8_3_expected)

ES_8_4_calculated = calculate_ES(data_7_1,0.05,0.94,method="Normal")
ES_8_4_expected = pd.read_csv("../testfiles/data/testout8_4.csv")
#print(ES_8_4_calculated)
#print(ES_8_4_expected)

ES_8_5_calculated = calculate_ES(data_7_2,0.05,0.94,method="T Distribution")
ES_8_5_expected = pd.read_csv("../testfiles/data/testout8_5.csv")
#print(ES_8_5_calculated)
#print(ES_8_5_expected)

ES_8_6_calculated = calculate_ES(data_7_2,0.05,0.94,method="historical")
ES_8_6_expected = pd.read_csv("../testfiles/data/testout8_6.csv")
#print(ES_8_6_calculated)
#print(ES_8_6_expected)

