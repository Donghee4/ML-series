#logistic regression 이진 분류 문제에서 클래스 확률을 예측
import pandas as pd
fish = pd.read_csv('https://bit.ly/fish_csv_data')
print(fish.head())
# print(pd.unique(fish['Species']))

fish_input = fish[['Weight']]