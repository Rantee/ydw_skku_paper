#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 14:02:10 2020

@author: mycelebs-it8
"""


import GKFN
import ft
import pandas as pd
import numpy as np

test = pd.read_excel('insurance_final.xlsx')
del test['Unnamed: 0']

all_X = test[['age', 'sex', 'bmi', 'children', 'smoker']].to_numpy()
all_Y = test['charges'].to_numpy()

trX = all_X[:-130]
teX = all_X[-130:]
trY = all_Y[:-130]
teY = all_Y[-130:]



# parameter를 설정하고 학습을 시킵니다.
alpha = 0.8
loop = 5
Kernel_Num = 10

# GKFN.GKFN(trX, trY, teX, teY, alpha, loop, Kernel_Num)



from sklearn.linear_model import LinearRegression
mlr = LinearRegression()
mlr.fit(trX, trY)

# The coefficients
print('Coefficients: \n', mlr.coef_)

# Intercept
print('Intercept : \n', mlr.intercept_)

# mean square error
print("MSE: %.2f" % np.mean((mlr.predict(teX) - teY) ** 2))

# r square
y_hat = mlr.predict(teX)
SSR = sum((teY - y_hat) ** 2)
SST = sum((teY - np.mean(teY)) ** 2)
r_squared = 1 - (float(SSR) / SST)
adj_r_squared = 1 - (1 - r_squared) * (len(teY) - 1) / (len(teY) - teX.shape[1] - 1)
print(r_squared, adj_r_squared)

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % mlr.score(teX, teY))