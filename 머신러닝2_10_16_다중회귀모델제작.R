
library(MASS)
#install.packages("car")
library(car)
#install.packages("skimr")
library(skimr)
#install.packages("caret")
library(caret)

data("Boston")
df <- Boston
skim(df)

initial_predictors <- c("lstat", "rm", "age", "dis", "rad", "tax", "ptratio", "indus", "nox",
                        "crim")

reusult <- lm(df$medv ~ ., data= df[,initial_predictors])
vif_values <- vif(result)

print(vif_value)

selected_predictors <- names(vif_values[vif_values < 7])
print(selected_predictors)

train_indices <- createDataPartition(df$medv, p = 0.8, list = FALSE)
df_train <- df[train_indices, ]
df_test <- df[-train_indices, ]

x_train <- df_train[, selected_predictors]
y_train <- df_train$medv

x_test <- df_test[, selected_predictors]
y_test <- df_test$medv

x_train <- cbind(1, as.matrix(x_train))
y_train <- as.numeric(y_train)

x_test <- cbind(1, as.matrix(x_test))
y_test <- as.numeric(y_test)

beta <- rep(0, ncol(x_train))

# 경사하강법 코드 7주차 압축파일에 존재
gradient_descent <- function(X, y, beta, learning_rate, num_iterations) {
  m <- nrow(X)
  cost_history <- numeric(num_iterations)
  for (i in 1:num_iterations) {
    y_pred <- X %*% beta
    residuals <- y_pred - y
    gradient <- t(X) %*% residuals / m
    beta <- beta - learning_rate * gradient
    cost <- sum(residuals^2) / (2 * m)
    cost_history[i] <- cost
    if (i %% 1000 == 0) {
      cat("Iteration", i, " | Cost:", cost, "\n")
    }
  }
  return(list(beta = beta, cost_history = cost_history))
}

#Hyper Parameters 정의
learning_rate <- 0.0001
num_iterations <- 10000

# 경사 하강법으로 도출한 Parameters 도출
gd_result <- gradient_descent(x_train, y_train, beta, learning_rate, num_iterations)
final_beta <- gd_result$beta
cost_history <- gd_result$cost_history

# 최종 베타값이 최종 가중치이다
# 경사 하강법 그래프
plot(1:num_iterations, cost_history, type = "l", col = "blue", x_lab = "Iteration", 
     ylab = "Cost", main = "Cost Function over Iterations")

# 행렬의 곱셈
y_pred <- X_test%*% final_beta

# 잔차 제곱합
ss_res <- sum((y_test-y_pred)^2)

# 총 제곱합
ss_total <- sum((y_test-mean(y_test))^2)

# 결정 계수
r2_test <- 1 -(ss_res/ ss_total)

# MSE 및 RMSE 계산
mse_test <- mse_function(final_beta, X_test, y_test)
rmse_test <- sqrt(mse_test)

print(paste("Mean Squared Error(MSE) on Test Set:", round(mse_test, 3)))
print(paste("Root Mean Squared Error(RMSE) on Test Set:", round(rmse_test, 3)))
print(paste("R-squared (R²) on Test Set:", round(r2_test,3)))

# pvalue, 검증, 변수선택이 빠져있다 왜?,
# 통계적 방법에는 다양한검증 + 변수선택(전진,후진,중위 등)
# 변수들에 대한 특징을 파악해서 학습을 하기 때문에 조금씩 다를 수 있다.
# 통계적인 방법으로 돌리면 값 동일 하지만 머신러닝으로 하면 값이 다르게 나온다
# 그래도 크게 범주를 벗어나지 않는다.
# 머신러닝 회귀
# 통계와 비교했을때 5개의 검증(선형성, 독립성...)가 없다 하지만 그것을 해주는 것이 결과 유추에 유리하다
# 모두 비교해주면 힘드니깐 vif값만 이용해서 추출을 해주었다
# 머신러닝이니깐 데이터를 학습80과 검증20으로 나눠야 한다
# 학습을 위해서 1이란 값을 넣어줬다 값이 회귀모델로 들어갈 때 mse function이
# cost fuction이 작아야 좋은 모델이다. 한번에 찾기 어렵고 이것은 학습이며
# 웨이트 값을 계속 줄여나가야 한다. 경사하강법을 추가해서 왔다리 갔다리 한다.
# 약 10000번의 학습을 한거고 계속 웨이트를 수정하고 어떤한 결과에 도달을 했다
# 베타값이 나온다 그 값을 검증데이터에 넣어서 y_preid을 구하고 실제 y을 빼서 검증을 한다 ?
# 이런 흐름만 이해하라 뒤에꺼 이해 못 한다. 모든 머신러닝, 인공지능 기법이
# 이것들에 기초를 해서 만들어졌다





