# SOURCE CODE______________________________________________________________________________________________________________

import numpy as np
from sklearn.linear_model import LinearRegression


def Predict(newdata, coefs):
    p = newdata.shape[1]
    betas = np.zeros(p, dtype = np.float16)
    beta0 = coefs[0]
    betas[:] = coefs[1:]
    Y_hat = newdata @ betas
    Y_hat = Y_hat + beta0

    return Y_hat


def MSE(y_true, y_pred):
    N = len(y_true)
    diff = np.zeros(N, dtype = np.float16)

    for i in range(0,N):
        diff[i] = (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i])
    
    mse = sum(diff)/N
    return(mse)


def BootstrapModel(X, Y, n_iter, alpha, location):
    n,p = X.shape
    coefs = np.zeros((n_iter, p+1), dtype = np.float16)
    data = np.zeros((n,p+1), dtype = np.float16)
    data[:, 0:-1] = X
    data[:, -1] = Y

    for i in range(0,n_iter):

        data_boot = data[np.random.choice(n, n, replace = True), :]
        model = LinearRegression()
        model.fit(data_boot[:, 0:-1], data_boot[:, -1])
        beta0 = model.intercept_
        betas = model.coef_
        coefs[i,0] = beta0

        for j in range(1,p+1):
            coefs[i,j] = betas[j-1]
    
    coefs_final = np.zeros(p+1, dtype = np.float16)

    for j in range(0,p+1):
        current_beta = coefs[:,j]
        stdev = np.std(current_beta)*alpha

        if location == 'mean':
            mean = np.mean(current_beta)
            sampled_beta = np.random.normal(loc=mean, scale=stdev, size=1)

        elif location == 'median':
            median = np.median(current_beta)
            sampled_beta = np.random.normal(loc=median, scale=stdev, size=1)

        coefs_final[j] = sampled_beta

    return coefs_final


def Optimum(X_train, Y_train, X_test, Y_test, alpha, location, n_iter, n_iter_per_model):

    p = X_train.shape[1]
    tab = np.zeros((n_iter, p+2), dtype = np.float16)

    for i in range(0, n_iter):

        coefs_boot = BootstrapModel(X = X_train, Y = Y_train, location = location, n_iter = n_iter_per_model, alpha = alpha)
        tab[i, 0:-1] = coefs_boot
        Y_pred = Predict(X_test, coefs_boot)
        mse = MSE(Y_test, Y_pred)
        tab[i, -1] = mse
    
    mse_col = tab[:, -1]
    argmin_ = np.argmin(mse_col)
    optim_params = tab[argmin_, 0:-1]

    return optim_params

# TESTING GROUND________________________________________________________________________________________________________________
'''
n_simul = 1000

diff_array = np.zeros(n_simul, dtype = np.float16)

for i in range(0, n_simul):

    print(str(np.round(i/n_simul*100, decimals=4)) + " %")
    N = 500
    P = 50
    X = np.random.normal(loc=0, scale=10, size=(N, P))
    betas_true = np.random.uniform(low = -10.0, high = 10.0, size = P+1)
    beta0 = betas_true[0]
    betas = betas_true[1:]
    Y = X @ betas
    Y = Y + beta0
    Y = Y + np.random.normal(loc = 0, scale = np.std(Y)*0.1, size=N)
    k = int(np.round(N*0.7))
    X_train = X[:k, :]
    X_test = X[k:, :]
    Y_train = Y[:k]
    Y_test = Y[k:]


    model_orig = LinearRegression()
    model_orig.fit(X_train, Y_train)
    coefs_orig = np.zeros(P+1, dtype = np.float16)
    coefs_orig[0] = model_orig.intercept_
    coefs_orig[1:] = model_orig.coef_
    Y_pred_orig = Predict(X_test, coefs_orig)
    MSE_orig = MSE(Y_test, Y_pred_orig)

    opt = Optimum(X_train, Y_train, X_test, Y_test, alpha = 0.01, location = 'mean', n_iter = 100, n_iter_per_model=100)

    Y_pred_boot_opt = Predict(X_test, opt)
    MSE_boot_opt = MSE(Y_test, Y_pred_boot_opt)

    diff = MSE_orig - MSE_boot_opt
    diff_array[i] = diff



np.savetxt('diff_array.txt', diff_array, delimiter=' ', fmt='%d')
print('all done.')
'''