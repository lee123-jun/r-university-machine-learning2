# R 최신버전으로 업데이트
install.packages("installr")
library(installr)
check.for.updates.R()
install.R()

install.packages("tidyverse") #데이터 구조 파악
install.packages("tidymodels") # 기계학습
install.packages("rstatix") # 통계 테스트
install.packages("GGally") # 복잡한 그래프 생성
install.packages("skimr") # 데이터의 흐름 요약

library(tidyverse)
library(tidymodels)
library(rstatix)
library(GGally)
library(skimr)

#데이터 불러오기
df <- read_csv("C:/Users/USER/Desktop/data/diabetes.csv", na = ".")
skim(df)

#문자형을 요소형으로 변환
#df$Outcome=as.factor(df$Outcome) #하나의 변수만 변환하는 방법
#df <- mutate_if(df, is.character, as.factor) # 모든 문자변수들을 전부 변환하는 방법

#불필요 번수 제거! value가 8개로 됨!
df <- select(df, -c(Outcome))
df <- select(df, -c(var1, var2, var3))
skim(df)

# 다중회귀 분석
result <- lm(df$Diabetes ~ ., data = df)
result1 <- lm(df$Diabetes ~ 1, data = df) # 다른 변수 고려 x

# 아노바 분석을 활용해 두 집단의 의미있는 차이가 있는지 확인
anova_result <- anova(result, result1)
print(anova_result)

# 분석 결과 확인
tidy_result <- tidy(result)
tidy_result <- mutate_if(tidy_result, is.numeric, round,3 )s
print(tidy_result)

#모델 요약
glance(result)

# 새로운 데이터를 csv에서 불러오기
new_data <- read.csv("C:/Users/USER/Desktop/data/diabetes_test.csv")

# 모델에 새로운 데이터 추가
predicted_values <- predict(result, newdata = new_data)

# 예측 결과 출력
print(predicted_values)

# 예측 결과를 새로운 CSV 파일로 저장 (옵션)
write.csv(predicted_values, "predicted_values.csv", row.names = FALSE)



