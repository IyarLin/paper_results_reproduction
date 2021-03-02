library(sgd)
beta <- runif(1000)
beta <- beta/sqrt(sum(beta^2)) ## ensure that the norm equals 1
M <- 200 ## number of simulations
N <- seq(995, 1005, by = 1) ## number of observations
test_MSE <- matrix(nrow = length(N), ncol = M)

for (i in 1:length(N)){
  for (m in 1:M){
    print(paste0("i=", i, ", m=", m))
    X <- replicate(1000, rnorm(N[i]))
    e <- rnorm(N[i], sd = 0.1)
    y <- X %*% beta + e
    dat <- as.data.frame(cbind(y, X))
    names(dat)[1] <- "y"
    mod <- sgd(formula = y ~. - 1, data = dat, model = "lm")
    
    X_test <- replicate(1000, rnorm(10000))
    e_test <- rnorm(10000, sd = 0.1)
    y_test <- X_test %*% beta + e_test
    
    preds_test <- predict(mod, X_test)
    test_MSE[i, m] <- sqrt(mean((y_test - preds_test)^2))
  }
}

plot(N, apply(test_MSE, 1, mean))
