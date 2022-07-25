#다중 회귀(multiple regression)
#인풋은 길이, 높이, 두께, 타켓은 무게
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np



df = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = df.to_numpy()

perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
        115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
        150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
        218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
        556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
        850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
        1000.0])
train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)

#이부분 다시보기-------------------------------------------------
#다중 인풋 처리

poly = PolynomialFeatures(include_bias = False) #절편항무시
poly.fit(train_input)
train_poly = poly.transform(train_input)
# print(train_poly.shape)
# print(poly.get_feature_names())
test_poly = poly.transform(test_input)  #테스트 폴리(테스트 인풋용)
#---------------------------------------------------------------

lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(test_poly, test_target))