#다중 분류 문제
#logistic regression 이름은 회귀지만 분류 모델임
#이진 분류 문제에서 클래스 확률을 예측
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

fish = pd.read_csv('https://bit.ly/fish_csv_data')
# print(fish.head()) #데이터와 특성 확인
# print(pd.unique(fish['Species']))   #물고기 종류 개수 확인

fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
fish_target = fish['Species'].to_numpy()
# print(fish_input[:5])
# print(fish_target[:5])

train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state= 42)
ss = StandardScaler()   #훈련 세트, 테스트 세트의 표준화 전처리
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

#---------------------------------------------------
# #최근접이웃을 이용한 확률 예측
# #주변 이웃의 개수에 따라서 비율을 정하므로 확률이라 하기에 어려움

# kn = KNeighborsClassifier(n_neighbors=3)
# kn.fit(train_scaled, train_target)
# # print(kn.score(train_scaled, train_target)) #0.89075    
# # print(kn.score(test_scaled, test_target))   #0.85

# # print(kn.classes_)    #순서대로 'Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish' 로 저장됨
# print(kn.predict(test_scaled[:5]))
# proba = kn.predict_proba(test_scaled[:5])
# print(np.round(proba, decimals=4))  #각 클래스에 속할 확률(여기서는 이웃 데이터들의 클래스 비율을 나타냄)
# print(test_target[:5])

# #test_scaled 의 4번째 데이터의 정답은 whitefish지만 모델의 답은 perch라고 오답을 낸다

# distances, indexes = kn.kneighbors(test_scaled[3:4])
# print(train_target[indexes])    #이웃 데이터의 답은 [['Roach' 'Perch' 'Perch']] 셋 모두 정답인 Whitefish 와 다름
#-----------------------------------------------------

#최근접이웃을 사용하지 않고 각 입력값에 multiplier를 적용한 z을 구하고, 시그모이드 함수를 이용해 확률을 구함
