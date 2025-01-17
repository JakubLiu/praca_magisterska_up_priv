#!/usr/bin/Rscript

source("/media/DANE/home/jliu/MASTER_THESIS/overfitting_remedy/mod.R")

# paramter over which we iterate
train_sizes <- seq(from = 10, to = 1000, by = 10)

# constants
p <- 5
coefs_true <- runif((p+1), -5, 5)
n_test <- 100
alpha_param <- 1.0
location_param <- 'median'

# output table
tab <- data.frame(matrix(rep(0,length(train_sizes)*3),
                         nrow = length(train_sizes),
                         ncol = 3))
colnames(tab) <- c("SampleSize", "median", "var")

for(i in 1:length(train_sizes)){

  status <- paste0(round(i/length(train_sizes)*100), " %")
  print(status)


  # create training data of given size
  n_train <- train_sizes[i]
  X_train <- matrix(rnorm(n_train*p, 0, 1), nrow = n_train, ncol = p)
  Y_train <- X_train%*%coefs_true[2:(p+1)]
  Y_train <- Y_train + coefs_true[1]
  Y_train <- Y_train + rnorm(n_train, 0, sd(Y_train)*0.5)

  # create test data
  X_test <- matrix(rnorm(n_test*p, 0, 1), nrow = n_test, ncol = p)
  Y_test <- X_test%*%coefs_true[2:(p+1)]
  Y_test <- Y_test + coefs_true[1]
  Y_test <- Y_test + rnorm(n_test, 0, sd(Y_test)*0.5)


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

  tab$SampleSize[i] <- train_sizes[i]
  tab$median[i] <- mean(diffs)
  tab$var[i] <- var(diffs)
}


path_out <- "/media/DANE/home/jliu/MASTER_THESIS/overfitting_remedy/SYSTEMATIC_SIM/Sample_Size_sim_res_median.txt"
write.csv(tab, path_out, row.names = FALSE)
print('done.')

