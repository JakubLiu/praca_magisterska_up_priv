#!/usr/bin/Rscript

source("/media/DANE/home/jliu/MASTER_THESIS/overfitting_remedy/mod.R")

# this function adds a noise of a greater magnitude to y's with higher values
# the hetero_power parameter controls the power of the heteroscedasticity (higher parmeter values imply higher heteroscedasticity)
hetero <- function(y, hetero_power){
    W <- sample(c(-1,1),1,replace=T)
    noise_mean <- abs(y)*hetero_power*W
    noise <- rnorm(1,noise_mean,1)
    y_new <- y + noise
    return(y_new)
}


# parameters over which we iterate
hetero_pwr <- seq(from = 0.1, to = 5.0, by = 0.01)

# constants
p <- 5
n <- 100
alpha_param <- 1.0
location_param <- 'mean'

tab <- data.frame(matrix(rep(0, length(hetero_pwr)*3),
                            nrow = length(hetero_pwr),
                            ncol = 3))

colnames(tab) <- c("Hetero_pwr", "mu", "var")

for(i in 1:length(hetero_pwr)){

    status <- paste0(i/length(hetero_pwr)*100, " %")
    print(status)

    # create training data of given size
    hetero_pwr_i <- hetero_pwr[i]
    coefs_true <- runif((p+1), -5, 5)
    X_train <- matrix(rnorm(n*p, 0, 1), nrow = n, ncol = p)
    Y_train <- X_train%*%coefs_true[2:(p+1)]
    Y_train <- Y_train + coefs_true[1]
    Y_train <- hetero(Y_train, hetero_power = hetero_pwr_i) # add heteroscedastic noise

    # create test data
    X_test <- matrix(rnorm(n*p, 0, 1), nrow = n, ncol = p)
    Y_test <- X_test%*%coefs_true[2:(p+1)]
    Y_test <- Y_test + coefs_true[1]
    Y_test <- hetero(Y_test, hetero_power = hetero_pwr_i) # add heteroscedastic noise


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

    tab$Hetero_pwr[i] <- hetero_pwr_i
    tab$mu[i] <- mean(diffs)
    tab$var[i] <- var(diffs)
}

path_out <- "/media/DANE/home/jliu/MASTER_THESIS/overfitting_remedy/SYSTEMATIC_SIM/HETEROSCEDASTICITY/Hetero_mean.txt"
write.csv(tab, path_out, row.names = FALSE, col.names = TRUE)
print('done.')