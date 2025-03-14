# 머신러닝2 앙상블 전 시간에 랜덤포레스
# 잔차(알수없는 오류값)값이 너무 크다 -> 잔차를 줄여야한다 -> 새로운 변수들 더 추가해야함
# 서로서로 장점을 통해서 문제를 해결한다
# 배깅과 부스팅에 대해서 잘 기억을 해라! / 랜덤포레스트(배깅), adaboost(부스팅)
# roc 커브 그리는 건 시험에 내지 않을 것이다...! decision tree는 시험 안냄
# roc 개념은 낼 수도 있지만, 코드는 시험에 안 낼 것이다. 다른 데이터도 동일하게 사용할 수 있다.
# 로지스틱회귀분석부터 시험 -> 시그모이드 함수 사용 2가지 집단 확률적으로 분류를 하였다 잘 이해
# 회귀모델과 분류모델에 있어서 회귀모델에는 어떻게 결론 도출? -> r^2 설명력으로 도추 
# 범주형에서는 roc 커브에는 정확도와 리콜 값이 있다. 이 두개를 이용해서 roc 커브를 만듣었다
# 뉴럴 네크워크를 배움 핵심은 퍼셉트론이란 것을 이어서 붙여보자 레이어를 다중 퍼셉트론
# 이런 xor 문제도 해결 가능해짐, 프로세스를 배웠다 가장 중요한건 함성함수, 활성화함수가 가장 중요했었다
# 시그모이드 함수르 이용해서 활성화함수를 구축했다. 하지만 다양한 문제점으로 인해 
# 보완하기 위해ㅓㅅ relu나 leak 랠로 등등 방식이 나왔다. 역전파 -> 흐름을 파악을 했따
# 활성화함수를 통해 나온 결론을 왜 역전파하는가 ? 하이퍼 파라미터 에포크에 따른 러닝 레이트
# 서포트 벡터머신 두 집단을 분류함에 있어서 너무 이상적인 집단이 떨어져있을때 기존 로지스틱 회귀로
분류하면 문제가 생겨서 더 좋게 분류할 순 없을 까 ? 마진이란 초평면을 이용해서 연구가 시작되었다
마진을최대화 하는 것이다. 마진주변의 데이터만 활용해서 마진안에 들어온 데이터는 상용 x 마진 밖도 
사용하지 않을 것이다. 제약조건 모든 샘플이 바깥으로 있어야 한다. 안쪽으로 들오몀ㄴ안된다. -> 하드마진
유하게 바꾸자 -> 소프트 마진/ 다양한 제약조건을 넣어서 이것이 서포트 벡터머신의 하이퍼파라미터 튜닝
크사이 이런거 몰라도 됩니다. 코스트와 감마가 크고 작을 때 데이터의 특징 이거 외워라, 커널트리는 넘어가라
결정 트리 -> 가장 직관적으로 데이터를 분류하는 것을 해석할 수 있다 블랙박스, 화이트 박스 사용가능
뿌리 노드 부모노드 자식노드 최종노드 결정트리의 핵심이 기존에는 분류로 하였다면 다양한 조건으로 가지를 쳐준다
결정트리에서 의사결정나무는 분류도 가능 회귀도 가능하다 범주형 데이터가 많으면 분석을 잘 진행 할 수가 업삳
연속형은 계산 이지 하지만 범주형은 실질적으로 의사결정 나무는 조건으로 분류하는 것이기 때문에 잘 분류를 할 수 있다
가장 중요 불순도 -> 작으면 좋다 왜 ? 우리가 트리를 분류함에 있어서 분수인자가 많이 들어가면 분류가 잘 안됐다
조건을 통해서 분류를 할 때 분순도가 가장 작을 것을 찾는 것이 중요하다 cart 알고리즘-> 지니계수 이분류에 특화됨
cart알고리즘에는 엔트로피란 개념을 이용해서 정보이득 지수 너무 정규화가 되지 않아서 너뭄 많은 집단들로 나눠지게 되서
전혀의미가 없다. 이득율이란 값을 이용해서 값을 잘 분류 시켰고, 높으면 높을 수록 좋다
가지가 너무 많은면 과적합으로 인해 -> 가지치기를 진행한다 이 스텝만 잘 알고 있어라.
보팅과 부스스트램(데이터를 샘플로 뽑아서 새로운 데이터셋을 만드는 것)ㅡㅇㄹ 이용해서 데이터 뽑아서
활용해서 동일한 모델을 써서 투표를 진행한다! 배깅에는 문제점이 존재 하지만 문제점잉이있어
보완위해 ㅇ약한 분류기 부스팅이란 개념을 만들었다. 부스팅은 너무 많은 모델을 붙이게 되면 오버피팅이 
가능하다. 어떻게 가중치를 주냐. 부스팅에 대한 개념만 잘 이해해라! 트리모델의 정확도 그래프, 지니계수 그래프
그만큼의 영향력이 있다. 변수선택에도 잘 이용할 수 있다. 

# Randomforest
install.packages("randomForest")
library(randomForest)
library(caret)
df=read.csv("C:/Users/hj123/Desktop/학기중파일/학기중파일/2024_2학기 수업/수_머신러닝2/customer.csv")
df=na.omit(df) #결측치가 없어야함
df$Segmentation <- as.factor(df$Segmentation)

# 데이터 분할 
set.seed(123)
train_indices <- createDataPartition(df$Segmentation, p = 0.8, list = FALSE)
train_data <- df[train_indices, ]
test_data <- df[-train_indices, ]
train_data[sapply(train_data, is.character)] <- lapply(train_data[sapply(train_data, is.character)], as.factor)
test_data[sapply(test_data, is.character)] <- lapply(test_data[sapply(test_data, is.character)], as.factor)

# Hyper parameters 설정 및 모델
tree_model <- randomForest(Segmentation ~ ., data = train_data, ntree = 100, mtry = 2, importance = TRUE)

# ntree : 만들 개별 트리의 개수
# mtry : 노드 분할 시 고려할 변수의 개수 지정
# 모델 결과
summary(tree_model)
varImpPlot(tree_model) #변수 중요도 평가
predicted_probs <- predict(tree_model, test_data, type = "prob")
predicted_probs
predicted_classes <- colnames(predicted_probs)[apply(predicted_probs, 1, which.max)]
predicted_classes <- as.factor(predicted_classes)
predicted_classes
confusionMatrix(predicted_classes, test_data$)

# Adaboosting
install.packages("adabag")
library(adabag)
tree_model <- boosting(Segmentation ~ ., data = train_data, boos = TRUE, mfinal = 50)
predicted_probs <- predict(tree_model, test_data, n.trees = 100, type = "response")
# 모델 결과
summary(tree_model)
predicted_probs <- predict(tree_model, test_data, type = "prob")
predicted_probs
predicted_classes <- as.factor(predicted_probs$class)
predicted_classes
confusionMatrix(predicted_classes, test_data$Segmentation)

# 확률 정보 추출
if ("prob" %in% names(predicted_probs)) {
  predicted_probs <- predicted_probs$prob
}
# 확률 데이터가 행렬인지 확인하고 클래스 이름 설정
if (is.null(colnames(predicted_probs))) {
  colnames(predicted_probs) <- levels(test_data$Segmentation)
}

library(pROC)
# ROC 곡선 및 AUC 계산
roc_curves <- list()
auc_values <- numeric()
for (class in levels(test_data$Segmentation)) {
  binary_labels <- ifelse(test_data$Segmentation == class, 1, 0)
  class_probs <- predicted_probs[, class] # 클래스별 확률 추출
  roc_curves[[class]] <- roc(binary_labels, class_probs, levels = c(0, 1))
  auc_values[class] <- auc(roc_curves[[class]])
}
# ROC 곡선 플롯
par(mfrow = c(2, 2)) # 클래스의 수만큼 그래프 그리기
for (class in names(roc_curves)) {
  plot(roc_curves[[class]], main = paste("ROC Curve for Class:", class))
}
# AUC 값 출력
auc_values




