#!/usr/bin/Rscript

source("/media/DANE/home/jliu/MASTER_THESIS/overfitting_remedy/mod.R")

# parameters over which we iterate
noise_perc <- seq(from = 0.1, to = 2.0, by = 0.01)

# constants
p <- 5
n <- 100
alpha_param <- 1.0
location_param <- 'median'

tab <- data.frame(matrix(rep(0, length(noise_perc)*3),
                            nrow = length(noise_perc),
                            ncol = 3))

colnames(tab) <- c("Noise_perc", "mu", "var")

for(i in 1:length(noise_perc)){

    status <- paste0(1/length(noise_perc)*100, " %")
    print(status)

    # create training data of given size
    noise_perc_i <- noise_perc[i]
    coefs_true <- runif((p+1), -5, 5)
    X_train <- matrix(rnorm(n*p, 0, 1), nrow = n, ncol = p)
    Y_train <- X_train%*%coefs_true[2:(p+1)]
    Y_train <- Y_train + coefs_true[1]
    Y_train <- Y_train + rnorm(n, 0, sd(Y_train)*noise_perc_i)

    # create test data
    X_test <- matrix(rnorm(n*p, 0, 1), nrow = n, ncol = p)
    Y_test <- X_test%*%coefs_true[2:(p+1)]
    Y_test <- Y_test + coefs_true[1]
    Y_test <- Y_test + rnorm(n, 0, sd(Y_test)*noise_perc_i)


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

    tab$Noise_perc[i] <- noise_perc_i
    tab$mu[i] <- mean(diffs)
    tab$var[i] <- var(diffs)
}

path_out <- "/media/DANE/home/jliu/MASTER_THESIS/overfitting_remedy/SYSTEMATIC_SIM/NOISE/Noise_median.txt"
write.csv(tab, path_out, row.names = FALSE, col.names = TRUE)
print('done.')