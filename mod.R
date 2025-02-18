Predict <- function(newdata, coefs){
    p <- ncol(newdata) + 1
    betas <- 1:(p-1)
    beta0 <- coefs[1]

    for(i in 2:p){
        betas[i-1] <- coefs[i]
    }
    
    Y_hat <- newdata %*% betas
    Y_hat <- Y_hat + beta0
    return(Y_hat)
}

MSE <- function(y_true, y_pred){
    res <- sum((y_true-y_pred)^2)
    res <- res/length(y_true)
    return(res)
}

Bootstrap_Model <- function(X,Y,n_iter,alpha,print_progress = FALSE, location){
    p <- ncol(X)
    n <- nrow(X)
    coefs <- matrix(rep(0,n_iter*(p+1)), nrow = n_iter, ncol = (p+1))
    data <- cbind(X,Y)

    # get coefficient estimates from bootstrap samples

    # loop with progress print (the if statement must be outside of the loop)
    if(print_progress == TRUE){
        for(i in 1:n_iter){

            print(paste0(round(i/n_iter,3)*100.00, "%"))
            data_boot <- data[sample(1:nrow(data), nrow(data), replace = T),]
            model_boot <- lm(data_boot[,ncol(data_boot)] ~ data_boot[,1:(ncol(data_boot)-1)])
            coefs_boot <- summary(model_boot)$coefficients

            for(j in 1:(p+1)){
                coefs[i,j] <- coefs_boot[j]
            }
        }
    }

    else if(print_progress == FALSE){
        for(i in 1:n_iter){
            data_boot <- data[sample(1:nrow(data), nrow(data), replace = T),]
            model_boot <- lm(data_boot[,ncol(data_boot)] ~ data_boot[,1:(ncol(data_boot)-1)])
            coefs_boot <- summary(model_boot)$coefficients

            for(j in 1:(p+1)){
                coefs[i,j] <- coefs_boot[j]
            }
        }
    }

    # sample estimates
    coefs_final <- matrix(1:(p+1),nrow = (p+1), ncol = 1)

    for(j in 1:(p+1)){
        if(location == 'mean'){mu <- mean(coefs[,j])}
        else if(location == 'median'){mu <- median(coefs[,j])}
        sigma <- sd(coefs[,j])*alpha
        sampled_coef <- rnorm(1,mu,sigma)
        coefs_final[j,] <- sampled_coef
    }
    return(coefs_final)
}


JackModel <- function(X, Y, alpha, location){

    p <- ncol(X)
    n <- nrow(X)
    coefs <- matrix(rep(0,n*(p+1)), nrow = n, ncol = (p+1))
    data <- cbind(X,Y)

    for(i in 1:n){
        data_jack <- data[-i,]
        coefs_jack <- summary(lm(data_jack[,p] ~ data_jack[,1:(p-1)]))$coefficients

        for(j in 1:(p+1)){
                coefs[i,j] <- coefs_jack[j]
        }
    }

    # sample estimates
    coefs_final <- matrix(1:(p+1),nrow = (p+1), ncol = 1)

    for(j in 1:(p+1)){
        if(location == 'mean'){mu <- mean(coefs[,j])}
        else if(location == 'median'){mu <- median(coefs[,j])}
        sigma <- sd(coefs[,j])*alpha
        sampled_coef <- rnorm(1,mu,sigma)
        coefs_final[j,] <- sampled_coef
    }

    return(coefs_final)
}