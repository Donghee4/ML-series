#다중 분류 문제
#logistic regression 이진 분류 문제에서 클래스 확률을 예측
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

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

#-----------------------------------
#최근접이웃을 이용한 확률 예측
#주변 이웃의 개수에 따라서 확률을 정함
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled, train_target)) #0.89075    
print(kn.score(test_scaled, test_target))   #0.85
#----------------------------
#다중 분류 문제. 확률이 100%이지만 전혀 다른 예측(오답)이 나올 수 있을 수 있다