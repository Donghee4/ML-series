#다중 회귀(multiple regression)
#인풋은 길이, 높이, 두께, 타켓은 무게
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
print(lr.score(train_poly, train_target))  #0.99031
print(lr.score(test_poly, test_target))    #0.97145   길이만 사용했을 때 나온 과소적합 문제가 사라짐

#------------------------------------------------------------------
#특성 5제곱과, 과대적합 방지를 위한 규제

poly = PolynomialFeatures(degree = 5, include_bias = False) #절편항무시
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)  

ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)
print(train_poly.shape)     #특성이 55개
#---------------------------------------------------------
#릿지 회귀
ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target)) #0.98961
print(ridge.score(test_scaled, test_target))   #0.97906

#하이퍼 파라미터 찾기
# train_score = []
# test_score = []
# alpha_list = [0.001, 0.01, 0.1, 1, 10]
# for alpha in alpha_list:
#     ridge = Ridge(alpha=alpha)
#     ridge.fit(train_scaled, train_target)
#     train_score.append(ridge.score(train_scaled, train_target))
#     test_score.append(ridge.score(test_scaled, test_target))

# plt.plot(np.log10(alpha_list), train_score)
# plt.plot(np.log10(alpha_list), test_score)
# plt.xlabel('alpha')
# plt.ylabel('R^2')
# plt.show()
#train은 충분히 높고,  test값이 가장 높은 10^-1 = 1/10 이 으로 결정함

ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))  #0.99038
print(ridge.score(test_scaled, test_target))    #0.98279
#--------------------------------------------------------------
#라쏘 회귀

lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))  #0.98978
print(lasso.score(test_scaled, test_target))    #0.98005

#하이퍼 파라미터 찾기
# train_score = []
# test_score = []
# alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
# for alpha in alpha_list:
#     lasso = Lasso(alpha=alpha, max_iter=10000)
#     lasso.fit(train_scaled, train_target)
#     train_score.append(lasso.score(train_scaled, train_target))
#     test_score.append(lasso.score(test_scaled, test_target))

# plt.plot(np.log10(alpha_list), train_score)
# plt.plot(np.log10(alpha_list), test_score)
# plt.xlabel('alpha')
# plt.ylabel('R^2')
# plt.show()
#트레인은 10^1까지 충분히 높고 테스트 또한 10^1 일 경우 가장 높으므로 10^1로 결정함

lasso = Lasso(alpha = 10)
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))  #0.98888
print(lasso.score(test_scaled, test_target))    #0.98244
#라쏘 모델 또한 릿지처럼 적절하게 규제하는 것이 확인됨

print(np.sum(lasso.coef_ == 0)) #55개중 40개가 0이므로 특성 15개만 고려함