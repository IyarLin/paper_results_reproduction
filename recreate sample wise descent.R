library(MASS)
beta <- runif(1000)
beta <- beta/sqrt(sum(beta^2)) ## ensure that the norm equals 1
M <- 50 ## number of simulations
N <- c(2, seq(100, 800, 100), seq(900, 990, 10), seq(991,1000,1), 
       seq(1001, 1009, 1), seq(1010, 1100, 10), seq(1200, 2000, 100)) ## number of observations
test_MSE <- matrix(nrow = length(N), ncol = M)

for (i in 1:length(N)){
  for (m in 1:M){
    print(paste0("n=", N[i], ", m=", m))
    X <- replicate(1000, rnorm(N[i]))
    e <- rnorm(N[i], sd = 0.1)
    y <- X %*% beta + e
    
    if (N[i] < 1000){
      beta_hat <- ginv(X) %*% y
    } else {
      dat <- as.data.frame(cbind(y, X))
      names(dat)[1] <- "y"
      lm_model <- lm(y ~ .-1, data = dat)
      beta_hat <- matrix(lm_model$coefficients, ncol = 1)
    }
    
    X_test <- replicate(1000, rnorm(10000))
    e_test <- rnorm(10000, sd = 0.1)
    y_test <- X_test %*% beta + e_test
    
    preds_test <- X_test %*% beta_hat
    test_MSE[i, m] <- sqrt(mean((y_test - preds_test)^2))
  }
}


plot(N, apply(test_MSE, 1, mean), type = "l")
