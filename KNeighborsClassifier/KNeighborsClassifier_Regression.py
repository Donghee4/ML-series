#k-최근접 이웃 회귀(K-Nearest Neighbors Regression)를 이용한 무게 추정
#추정값은 이웃 데이터의 타겟값의 평균으로 결정된다
#데이터는 농어의 길이, 무게
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
        21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
        23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
        27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
        39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
        44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
        115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
        150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
        218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
        556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
        850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
        1000.0])

#데이터 분리 및 차원 처리
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)
train_input = train_input.reshape(-1,1) #sklearn은 훈련 세트가 2차원이어야 하므로 데이터를 2차원으로 변경
test_input = test_input.reshape(-1,1)
# print(train_input.shape, test_input.shape)

knr = KNeighborsRegressor()
knr.fit(train_input, train_target)
print(knr.score(test_input, test_target))  #0.99280
#정확도는 R^2(결정계수)값으로 측정됨

test_prediction = knr.predict(test_input)
print(test_prediction)
mae = mean_absolute_error(test_target, test_prediction) #평균적으로 19g 차이
print(knr.score(train_input, train_target)) #0.96988    테스트 세트보다 훈련 세트가 점수가 낮으니 과소적합. 

knr.n_neighbors = 3     #참조하는 데이터의 수를 줄여서 그 특징을 더 잘 뽑아냄. 모델이 더 복잡해짐
knr.fit(train_input, train_target)
print(knr.score(train_input, train_target)) #0.98049
print(knr.score(test_input, test_target))   #0.97464    #테스트와 훈련 세트 점수 둘 다 높으므로 훈련이 잘 됨

# plt.scatter(perch_length, perch_weight)
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()
