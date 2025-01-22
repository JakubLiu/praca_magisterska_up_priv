import sys
sys.path.append('/media/DANE/home/jliu/MASTER_THESIS/overfitting_remedy/OPTIMUM/')
import mod as Opt
import numpy as np
from sklearn.linear_model import LinearRegression

# parameters over which we iterate
param_of_interest = list(range(10,1000,5))

# parameters held constants
n_iter_per_model_param = 1000
location_param = 'median'
n = 20
p = 5

tab = np.zeros((len(param_of_interest),3))  # colnames = [param_o_i, mu, var]  (mu, var of the dist of diff)

for i in range(0, len(param_of_interest)):

    status = str(i/len(param_of_interest)*100) + " %"
    print(status)
    n_iter_i = param_of_interest[i]
    
    # generate train dataset
    X_train = np.random.normal(loc=0, scale=1, size=(n,p))
    betas_true = np.random.uniform(low = -5.0, high = 5.0, size = p+1)
    beta0 = betas_true[0]
    betas = betas_true[1:]
    Y_train = X_train @ betas
    Y_train = Y_train + beta0
    Y_train = Y_train + np.random.normal(loc = 0, scale = np.std(Y_train)*0.1, size=n)

    # create test data
    X_test = np.random.normal(loc=0, scale=1, size=(n,p))
    Y_test = X_test @ betas
    Y_test = Y_test + beta0
    Y_test = Y_test + np.random.normal(loc = 0, scale = np.std(Y_test)*0.1, size=n)

    # fit and evaluate the stadatd model
    model_orig = LinearRegression()
    model_orig.fit(X_train, Y_train)
    coefs_orig = np.zeros(p+1, dtype = np.float16)
    coefs_orig[0] = model_orig.intercept_
    coefs_orig[1:] = model_orig.coef_
    Y_pred_orig = Opt.Predict(X_test, coefs_orig)
    MSE_orig = Opt.MSE(Y_test, Y_pred_orig)

    # fit and evaluate the optimum bootstrap model n = 1000 times
    n_opt_model_runs = 1000
    diffs = np.zeros(n_opt_model_runs, dtype = np.float16)

    for j in range(0,n_opt_model_runs):
        opt = Opt.Optimum(X_train, Y_train, X_test, Y_test, alpha = 1.0,
                          location = location_param, n_iter = n_iter_i,
                          n_iter_per_model = n_iter_per_model_param)
        Y_pred_boot_opt = Opt.Predict(X_test, opt)
        MSE_boot_opt = Opt.MSE(Y_test, Y_pred_boot_opt)
        diff = MSE_orig - MSE_boot_opt
        diffs[j] = diff
    

    tab[i,0] = n_iter_i
    tab[i,1] = np.mean(diffs)
    tab[i,2] = np.var(diffs)

output_file_path = '/media/DANE/home/jliu/MASTER_THESIS/overfitting_remedy/SYSTEMATIC_SIM/OPTIMUM_SIM/N_ITER/N_iter_sim_median.csv'
np.savetxt(output_file_path, tab, delimiter=',')
print('all done.')

