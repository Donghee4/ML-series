# ML-series
machine learning + deep learning <br>
1. KNeighborsClassifier - 분류를 위한 K_Nearest Neighbors 를 위한 추정 <br>
2. Regression - 목적 변수 추정을 위한 회귀
- 선형회귀 linear Regression
- 다항 회귀(polynomial regression)
- 다중 회귀(multiple regression) (여러 인풋을 사용함) + 규제 모델(ridge, lasso) 적용

3. probability and SDG - 분류 문제에서 확률 예측과 경사 하강법<br>
- logistic regression (회귀가 아닌 분류 모델) - 이진 및 다중 분류에서 확률을 예측함
    z 값을 기준으로 이진분류 -> sigmoid 함수 사용.<br>
    다중 분류 -> softmax 함수 사용으로 확률을 결정함.<br>
- Stochastic Gradient Decent - 추가되는 데이터를 경사 하강법을 이용해 모델을 훈련하는 방법<br>