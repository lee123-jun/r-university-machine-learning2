#2024.10.02 머신러닝2_5주차_김창균교수님

install.packages("car")
library(car)

# 잔차
opar <- par(no.readonly =  TRUE) # 그래프가 듣어갈 Window 생성
par(mfro = c(2,2)) # 그래프가 들어갈 공간 생성 2*2
plot(result) # 그래프 결과 도출
par(opar) # Restore the original graphical parameters

# 정규성: p>0.05보다 크면 귀무가설 채택 (정규성이 있음)
shapiro.test(result$residuals)

# 등분산성: p>0.05보다 크면 귀무가설 채택 (등분산성임!)
ncvTest(result)

durbinWatsonTest(result)

vif(result)

influenceplot(result, id.method="identify")

df = df[-c(121), ]
df = df[-c(121, 130, 145, 160), ] # 여러 개의 값 제거

result <- lm(df$Diabetes ~ ., data = df)

# 회귀분석을 하자! 
# 데이터 -> 분산분석(두 회귀모델/ 다  넣은 것, 다 넣지 않은 것 비교!) -> 두 집단은 차이가 존재한다! (차이가 존재해야 회귀분석을한다)
# -> 이상치 검증(크게는 3가지, 세세하게는 5가지, 잔차(등분산성), 독립성, 정규성) -> 이상치 파악 및 제거 -> 회귀 
# -> 변수선택법 3가지 존재 1. 후진소거법 (데이터 다 들어가고 뒤에서 부터 없애는 것)
# 2. 전진소거법 (다 빠진 상태에서 하나씩 추가를 한다) 3. 단계적선택법 -> 회귀분석 진행
# 모두 따라가려하진 말구 유동적으로 진행하여라 (회귀분석에서 검증으로 이동)

# AIC값이 줄어들면 그 변수는 의미가 있음
AIC(Akaike information criterion)

result_bk = lm(df$Diabetes ~., data=df)
Df_fit_bk = stepAIC(result_bk, direction= "backward", trace = T)
# trace 하나씩 빼는 것 보여줌

# 단계적선택법
Df_fit_st = stepAIC(result_bk, direction = "both", trace = T)

# 전진선택법
result_fw = lm(df$Diabetes ~ 1, data=df)

Df_fit_fk = step(result_fw, direction= "forward", scope = (backward 변수 카피), trace=T)

# 기계학습(지도학습 , 다중회귀분석, 로지스틱분석)
# 회귀분석의 핵심은 잔차를 최소화하는 것 -> 잔차의 제곱의 합이 최소가 되는 선을 긋는 것
# 가설검정 train 70 / test 30 , 우리가 하는 것은 기계 회귀분석이 아닌 통계 회귀분석이다
# sse로 구하는 회귀분석은 한 가지 선밖에 나올 수 없다!
# MSE = 1/n 표준화(평균을 구함)를 시킬 때 
# 경사하강법 cost function Hmm...
# 데이터 (회귀모델) -> 결과 / 데이터 (회귀모델) -><- (가중치 업데이트)/ 계속 왔다갔다-> 결과
# epcho란 회귀모델 가중치 업데이트 왔다갔다 하는 것
# 회귀분석 -> 다양한 변수 -> 표준화(잔차) -> learning late, epcho, 시작위치(조정을 하이퍼 파라미터 뉴닝!) 
# -> 최소 MSE -> 최소의 cost를 찾는 것이다.
# 데이터 -> MSE -> W값들이 나온다 -> learning late값들을 정한다 최적의 weight들을 조정을 한다
# 다음시간에는 전체적인 과정을 실습을 통해 알아본다!