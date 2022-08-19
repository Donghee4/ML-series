# classification algorithms
**다중 분류**

Logistic_regression - 다중 분류 문제. 다중 인풋에서 z 값을 구하고 softmax로 확률을 구함. 이진 분류의 경우 sigmoid로 확률을 구함<br>
로지스틱 회귀는 회귀가 아닌 분류 모델.<br>
클래스의 개수만큼 모델을 훈련함<br>
z = a * x1 + b * x2.... + c<br>
각 입력값과 가중치 곱에 상수를 더한 값으로 z 이 결정된다.<br>
z을 이용해 sigmoid 또는 softmax로 각 확률을 구함.<br>
클래스 분류시 가장 높은 확률을 가진 클래스로 분류함.<br>

Stochastic gradient decent - 훈련을 마친 후 데이터가 지속적으로 들어올 시, 확률적 경사 하강법을 이용해 추가 데이터를 학습함.<br>
여기서는 이진 분류 문제를 해결함