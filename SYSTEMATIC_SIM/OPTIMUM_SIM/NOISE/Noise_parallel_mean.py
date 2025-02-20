import sys
sys.path.append('/media/DANE/home/jliu/MASTER_THESIS/overfitting_remedy/OPTIMUM/')
import mod as Opt
import numpy as np
from sklearn.linear_model import LinearRegression
import multiprocessing as mp


# ===================================== functions for parallel computing =============================================================

# this function calculates the diff (MSE_orig - MSE_boot_opt)
def CalcDiff(j, X_train, X_test, Y_train, Y_test, alpha, location, n_iter, n_iter_per_model, MSE_orig):

    opt = Opt.Optimum(X_train, Y_train, X_test, Y_test, alpha=alpha,
                      location=location, n_iter=n_iter,
                      n_iter_per_model=n_iter_per_model)

    Y_pred_boot_opt = Opt.Predict(X_test, opt)
    MSE_boot_opt = Opt.MSE(Y_test, Y_pred_boot_opt)
    diff = MSE_orig - MSE_boot_opt
    
    return diff

# this function runs the CalDiff function mutliple times in parallel, in order do get the array of the diff values
def CalcDiffArray(NumCores, n_opt_model_runs, X_train, X_test, Y_train, Y_test, alpha, location, n_iter, n_iter_per_model, MSE_orig):
    
    with mp.Pool(processes=NumCores) as pool:
        diffs = pool.starmap(CalcDiff, [(j, X_train, X_test, Y_train, Y_test, alpha, location, n_iter, n_iter_per_model, MSE_orig) 
                                           for j in range(n_opt_model_runs)])

    return np.array(diffs)

# ======================================================================================================================================




# parameters over which we iterate
param_of_interest = list(np.linspace(0.1,2.0,191))  # noise percentage

# parameters held constants
n = 500
p = 5
n_iter_extension = 100
n_iter_per_model_param = 100
location_param = 'mean'
num_cores = 60

print('running on {} cores'.format(num_cores))

tab = np.zeros((len(param_of_interest),3))  # colnames = [param_o_i, mu, var]  (mu, var of the dist of diff)

cnt = 0
for i in range(0, len(param_of_interest)):

    status = cnt/len(param_of_interest)*100
    print(status, flush = True)
    cnt = cnt + 1
    noise_i = param_of_interest[i]
    
    # generate train dataset
    X_train = np.random.normal(loc=0, scale=1, size=(n,p))
    betas_true = np.random.uniform(low = -5.0, high = 5.0, size = p+1)
    beta0 = betas_true[0]
    betas = betas_true[1:]
    Y_train = X_train @ betas
    Y_train = Y_train + beta0
    Y_train = Y_train + np.random.normal(loc = 0, scale = np.std(Y_train)*noise_i, size=n)

    # create test data
    X_test = np.random.normal(loc=0, scale=1, size=(n,p))
    Y_test = X_test @ betas
    Y_test = Y_test + beta0
    Y_test = Y_test + np.random.normal(loc = 0, scale = np.std(Y_test)*noise_i, size=n)

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

    # fit and evaluate the optimum bootstrap model n = 1000 times
    diffs = CalcDiffArray(
                    NumCores=num_cores,
                    n_opt_model_runs=1000,
                    X_train=X_train,
                    X_test=X_test,
                    Y_train=Y_train,
                    Y_test=Y_test,
                    alpha=1.0,
                    location=location_param,
                    n_iter=n_iter_extension,
                    n_iter_per_model=n_iter_per_model_param,
                    MSE_orig=MSE_orig)
    

    tab[i,0] = noise_i
    tab[i,1] = np.mean(diffs)
    tab[i,2] = np.var(diffs)

output_file_path = '/media/DANE/home/jliu/MASTER_THESIS/overfitting_remedy/SYSTEMATIC_SIM/OPTIMUM_SIM/NOISE/Noise_sim_mean.csv'
np.savetxt(output_file_path, tab, delimiter=',')
print('all done.')

