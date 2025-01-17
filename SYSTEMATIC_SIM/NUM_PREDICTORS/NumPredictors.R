#!/usr/bin/Rscript

source("/media/DANE/home/jliu/MASTER_THESIS/overfitting_remedy/mod.R")

# parameters over which we iterate
p <- seq(from = 1, to = 100, by = 5)

# constants
n <- 500 # both for train and test
alpha_param <- 1.0
location_param <- 'median'

# output table
tab <- data.frame(matrix(rep(0,length(p)*3),
                         nrow = length(p),
                         ncol = 3))
colnames(tab) <- c("NumPredictors", "mu", "var")

for(i in 1:length(p)){

  status <- paste0(round(i/length(p)*100), " %")
  print(status)


  # create training data of given size
  num_p <- p[i]
  coefs_true <- runif((num_p+1), -5, 5)
  X_train <- matrix(rnorm(n*num_p, 0, 1), nrow = n, ncol = num_p)
  Y_train <- X_train%*%coefs_true[2:(num_p+1)]
  Y_train <- Y_train + coefs_true[1]
  Y_train <- Y_train + rnorm(n, 0, sd(Y_train)*0.5)

  # create test data
  X_test <- matrix(rnorm(n*num_p, 0, 1), nrow = n, ncol = num_p)
  Y_test <- X_test%*%coefs_true[2:(num_p+1)]
  Y_test <- Y_test + coefs_true[1]
  Y_test <- Y_test + rnorm(n, 0, sd(Y_test)*0.5)


  # fit and evaluate standard model
  model_orig <- lm(Y_train ~ X_train)
  coefs_orig <- summary(model_orig)$coefficients
  Y_hat_orig <- Predict(X_test, coefs_orig)
  MSE_orig <- MSE(Y_test, Y_hat_orig)


  # fit and evaluate bootstrap model n = 100 times
  diffs <- 1:1000
  for(j in 1:1000){
    coefs_boot <- Bootstrap_Model(X_train, Y_train, n_iter = 1000, alpha = alpha_param,
                                  location = location_param, print_progress = F)
    Y_hat_boot <- Predict(X_test, coefs_boot)
    MSE_boot <- MSE(Y_test, Y_hat_boot)
    diffs[j] <- MSE_orig - MSE_boot  # the larger the diff the better
  }

  tab$NumPredictors[i] <- p[i]
  tab$mu[i] <- mean(diffs)
  tab$var[i] <- var(diffs)
}


path_out <- "/media/DANE/home/jliu/MASTER_THESIS/overfitting_remedy/SYSTEMATIC_SIM/NUM_PREDICTORS/NumPredictors_median.txt"
write.csv(tab, path_out, row.names = FALSE, col.names = TRUE)
print('done.')
