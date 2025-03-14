gradient_descent <- function(X, y, beta, learning_rate, num_iterations) {
  m <- nrow(X)
  cost_history <- numeric(num_iterations)
  for (i in 1:num_iterations) {
    # 예측값 계산
    y_pred <- X %*% beta
    residuals <- y_pred - y
    gradient <- t(X) %*% residuals / m
    beta <- beta - learning_rate * gradient
    # 비용 함수 계산 및 저장
    cost <- sum(residuals^2) / (2 * m)
    cost_history[i] <- cost
    if (i %% 1000 == 0) {
      cat("Iteration", i, " | Cost:", cost, "\n")
    }
  }
  return(list(beta = beta, cost_history = cost_history))
}


mse_function <- function(beta, X, y) {
  # Calculate the predictions based on the current coefficients
  y_pred <- X %*% beta
  
  # Calculate the residuals (errors)
  residuals <- y_pred - y
  
  # Calculate the Mean Squared Error (MSE)
  mse <- sum(residuals^2) / nrow(X)
  
  # Return the MSE value
  return(mse)
}
