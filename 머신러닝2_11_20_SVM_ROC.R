# sigmode 함수를 사용해서 분류를 하였다
# 이상치에 대한 분류일 때 이상하게 분류되는 경우가 발생함
# 색깔이 없다면 어떻게 분류 딥러닝은 이상치에 대한 영향을 많이 받는다
# 이상치를 제외하고 분석을 하는 등의 프로세스를 거침
# 데이터를 전처리 할 때는 이상치를 제거한다 좀 더 쉽게 해결할 수 없을까 ?
# 회귀분석 -> 하나의 데이터를 가장 잘 표현할 수 있는 선을 그린것이다
# 서포트 벡터머신: 두 집단을 구분하기 위해서 선을 긋는 건 어떨까 ?
# 예시 피자-> 페퍼로니, 하와이안 종류별로 자른다 // 
# 하나의 선을 그려보자! 근데 그 선은 회귀에 대한 직선이 아니라 두 집단을 잘 나타낼 수 있는 선 ㄱ리자
# 집단을 구별하는 선을 그을 때 사람마다 미묘하게 다르게 그을 수가 있다
# 퍼셉트론의 확장된 개념 -> 서포트벡터머신 / 퍼셉트론이 모여서(다중 퍼셉트론) 신경망이 된다
# 서포트벡터머신 마진을 최대로 만든다.! -> 두 집단의 분류를 확실하게 하기 위해서
# 너무 동 떨어진 데이터 제외 마진에 가까운 데이터들만 사용한다
# W^T(전치행렬)*x + b = 0 
# 서포트 벡터의 가장 핵심 요소 마진범위 공식(x^+ = x^- + lamda*w) 이동의 폭이 람다
# 이 람다의 값 조절이 가장 중요하다 / p norm(벡터간의 거리를 계산하는 것)
# 제약조건 1. 마진 안에는 어떤한 데이터도 들어올 수 없다. hard margin 
# min*1/2||w|| 이 식을 이용하면 제약 조건에 위배되지 앟는다
# lagrangian multiplier (가중치 업데이트) 데이터 분포가 4함수일 시
# y -> 각각의 데이터 분포 , alpha -> learning rate (하이퍼 파라미터)이다.
# 크사이라는 개념, 하르마진에서는 마진 내 데이터가 들어오는 것을 허용하지 않지만
# 크사이라는 가중치, 하이퍼파라미터의 개념으로 마진 범위 내 들어오는 것을 허용한다.
# cost라는 하이퍼파라미터를 잘 조절하는 것이 가장 핵심 개념이다
# 커널트릭 데이터의 형태를 해지지 않는 선에서 차원을 바꿔주는 것
# 조금 더 쉽게 분류를 할 수 있도록 도와준다 -> gamma라는 값이다 
# cost와 gamma 값을 잘 조절해서 분류를 진행한다 
# cost가 크면 조금 더 유연하게 분류가능(곡선), gamma가 각각 많은 집단으로 분류하여
# overfitting 될 가능성이 크다.
# 서포트벡터머신은 분류문제에서 사용한다. (이상치가 많거나, 데이터가 꼬부랑지게 존재할때)
# 가장 핵심은 람다! 마진의 거리를 최대로 만드는 것
# 결국에는 소프트 벡터 머신이라는 거는 데이터들에 있어서 수직의 거리를 통해서
# 하나의 마진의 거리를 계산한다.

install.packages("e1071")
install.packages("caret")

library(e1071) # SVM 모델
library(caret)

count(df)
# 데이터 불러오기
df <- read.csv("C:/Users/hj123/Desktop/UniversalBank.csv")

# 종속변수에 대한 데이터 요소로 변환
df$Personal.Loan <- ifelse(df$Personal.Loan == 1, "High", "Low")
df$Personal.Loan <- as.factor(df$Personal.Loan)

#데이터 나누기 학습 80%, 테스트 20%
trainIndex <- createDataPartition(df$Personal.Loan, p = 0.8, list = FALSE)
train_data <- df[trainIndex,]
test_data <- df[-trainIndex,]

# 코스트, 감마 하이퍼 파라미터 튜닝
cost_value <- 1 # Regularization parameter C
gamma_value <- 0.1 # Gamma for RBF kernel

# SVM모델 생성
svm_model <- svm(train_data$Personal.Loan ~ ., data = train_data,
                 type = 'C-classification',
                 kernel = 'radial', #Radial Basis Function
                 cost = cost_value,
                 gamma = gamma_value,
                 probability = TRUE) #확률값으로 도출

summary(svm_model)
# number of support vectors: 595 왜 사람마다 다르냐 ?
# 통계와 머신러닝의 차이 8:2 비율로 랜덤으로 추출하여 분석을 했기 때문에
# 너무 멀리 있는 이상치가 날아가고, 마진 범위 데이터들 제외가 됐다
# summary 했으니깐 예측해봐야지, 확인하기 위해선 test 데이터를 사용해야함!
# 

type = 'one-classification', kernel = 'radial’ # One-classification(특정 클래스만 학습해 이상치 탐지
type = 'eps-regression', kernel = 'radial’ # eps-regression(연속형 데이터를 예측하는 회귀문제 사용)

#install.packages("pROC")
library(pROC)

predicted_probs <- predict(svm_model, test_data, probability = TRUE)
predicted_probs
predicted_probs <- attr(predicted_probs, "probabilities")[, "High"] #객체의 속성을 추출하는 함수
predicted_probs
predicted_classes <- ifelse(predicted_probs > 0.5, "High", "Low")
predicted_classes <- as.factor(predicted_classes)
predicted_classes
conf_matrix <- confusionMatrix(predicted_classes, test_data$Personal.Loan)
print(conf_matrix)
roc_curve <- roc(test_data$Personal.Loan, predicted_probs)
plot(roc_curve)
auc(roc_curve)

# 시험 칠때 대충 넘어간거는 그냥 대충 이해하고 넘어가고, 전반적인 흐름만 파악하고
# 넘어가라, 퍼셉트론 이애하고, roc 커브 이해하고 
# roc 커브 이해하기
# 1. True Positive Rate과 False Positive Rate / True: y축, false: x축
# 2. ROC Curve위의 한 점이 의미하는 것은 무엇인가? 
# 3. ROC Curve의 휜 정도가 의미하는 것은 무엇인가?
# , ROC 커브는 이진 분류기의 성능을 표현하는 커브이고, 
# 가능한 모든 threshold에 대해 FPR과 TPR의 비율을 표현한 것
# 이런 간단한 모델로 많이 효과가 있고 실무에서도 많이 사용한다
# 실제로는 간단한 알고리즘으로 구현이 많이 된다.


