#Stochastic Gradient Descent 확률적 경사 하강법
#SGD classifier를 이용해서 지속적으로 데이터가 들어올 때 추가로 학습하는 방법
#적절한 epoch 횟수를 찾아 과소, 과대적합이 일어나지 않을 숫자만큼만 훈련을 함
from random import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish_input = fish[['Weight', 'Length','Diagonal', 'Height','Width']].to_numpy()
fish_target = fish['Species'].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

ss = StandardScaler()   #데이터 세트를 표준화하는 전처리
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

#---------------------SGDClassifier 테스트 -----------------------------------------
# sc = SGDClassifier(loss = 'log_loss', max_iter=10, random_state=42)  #max_iter는 수행할 에포크 횟수, 샘플을 1개씩 꺼내는 확률적 경사 하강법
# sc.fit(train_scaled, train_target)
# print(sc.score(train_scaled, train_target)) #0.77310
# print(sc.score(test_scaled, test_target))   #0.775

# sc.partial_fit(train_scaled, train_target)  #추가 훈련
# print(sc.score(train_scaled, train_target)) #0.815126
# print(sc.score(test_scaled, test_target))   #0.85
# # 추가 훈련시 점수는 높아졌지만 이 이상 훈련을 하면 점수가 떨어진다
#-------------------------------------------------------------------------------------
#-----------------적절한 epoch 횟수 확인-----------------------------------------------
# sc = SGDClassifier(loss ='log_loss', random_state=42)
# train_score = []
# test_score = []
# classes = np.unique(train_target)

# for _ in range(0,300):
#     sc.partial_fit(train_scaled, train_target, classes = classes)
#     train_score.append(sc.score(train_scaled, train_target))
#     test_score.append(sc.score(test_scaled,test_target))

# plt.plot(train_score)
# plt.plot(test_score)
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.show()
# #과대적합이 일어나지 않는 100회 정도가 적당함
#--------------------------------------------------------------------------------------

sc = SGDClassifier(loss = 'log_loss', max_iter=100, tol=None, random_state=42)  #tol은 성능이 향상될 최솟값을 정하여 반복학습을 지속하게함. 여기서는 무조건 100회반복
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target)) #0.95798
print(sc.score(test_scaled, test_target))   #0.925