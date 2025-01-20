#!/usr/bin/Rscript
source("/media/DANE/home/jliu/MASTER_THESIS/overfitting_remedy/mod.R")
library(MASS)

# parameters over which we iterate
corr <- seq(from = 0.0, to = 1.0, by = 0.01)
print(length(corr))
# constants
p <- 5
n <- 100
alpha_param <- 1.0
location_param <- 'median'

tab <- data.frame(matrix(rep(0, length(corr)*3),
                            nrow = length(corr),
                            ncol = 3))

colnames(tab) <- c("Predictor_correlation", "mu", "var")

for(i in 1:length(corr)){

    status <- paste0(i/length(corr)*100, " %")
    print(status)

    # create training data of given size
    corr_i <- corr[i]
    coefs_true <- runif((p+1), -5, 5)
    means <- runif(p,-5,5)
    covariance_mat <- matrix(rep(corr_i,p*p), nrow = p, ncol = p)
    diag(covariance_mat) <- 1  # set the same correlation for all predictors

    X_train <- mvrnorm(n = n, mu = means, Sigma = covariance_mat)
    Y_train <- X_train%*%coefs_true[2:(p+1)]
    Y_train <- Y_train + coefs_true[1]
    Y_train <- Y_train + rnorm(n, 0, sd(Y_train)*0.1)

    # create test data
    X_test <- mvrnorm(n = n, mu = means, Sigma = covariance_mat)
    Y_test <- X_test%*%coefs_true[2:(p+1)]
    Y_test <- Y_test + coefs_true[1]
    Y_test <- Y_test + rnorm(n, 0, sd(Y_test)*0.1)


    # fit and evaluate standard model
    model_orig <- lm(Y_train ~ X_train)
    coefs_orig <- summary(model_orig)$coefficients
    Y_hat_orig <- Predict(X_test, coefs_orig)
    MSE_orig <- MSE(Y_test, Y_hat_orig)

    # fit and evaluate bootstrap model n = 1000 times
    diffs <- 1:1000
    for(j in 1:1000){
        coefs_boot <- Bootstrap_Model(X_train, Y_train, n_iter = 1000, alpha = alpha_param,
                                    location = location_param, print_progress = F)
        Y_hat_boot <- Predict(X_test, coefs_boot)
        MSE_boot <- MSE(Y_test, Y_hat_boot)
        diffs[j] <- MSE_orig - MSE_boot  # the larger the diff the better
    }

    tab$Predictor_correlation[i] <- noise_perc_i
    tab$mu[i] <- mean(diffs)
    tab$var[i] <- var(diffs)
}

path_out <- "/media/DANE/home/jliu/MASTER_THESIS/overfitting_remedy/SYSTEMATIC_SIM/PREDICTOR_CORRELATION/PredCorr_median.txt"
write.csv(tab, path_out, row.names = FALSE, col.names = TRUE)
print('done.')
