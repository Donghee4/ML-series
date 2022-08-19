#Stochastic Gradient Descent 확률적 경사 하강법
#SGD classsifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish_input = fish[['Weight', 'Length','Diagonal', 'Height','Width']].to_numpy()
fish_target = fish['Species'].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

ss = StandardScaler()   #데이터 세트를 표준화하는 전처리
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

print(fish_input.size)
f = fish['Weight'].to_numpy()
print(f.size)
