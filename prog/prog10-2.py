# -*- coding: utf-8 -*-
from numpy import *
from scipy import linalg as LA
from matplotlib.pyplot import *

# 多項式フィッティング

# 学習用データ
N = 10
train_x = random.uniform(0, 1, N)
train_y = random.normal(sin(2*pi*train_x), 0.1)

# 多項式モデルの次数
M = 10

# 正則化項の係数
LAMBDA = 0.001

# 基底関数
def phi(x):
    return array([x**k for k in range(M+1)])

# 計画行列
X = zeros((N, M+1))
for i in range(N):
    X[i,:] = phi(train_x[i])

# 学習 (ここでは X^TX正則性チェックは省略)
w = LA.solve(X.T.dot(X) + LAMBDA*identity(M+1), X.T.dot(train_y))

# 学習結果の表示
x = linspace(0, 1, 100)
y = zeros(100)
for i in range(100):
    y[i] = w.dot(phi(x[i]))

xlim(0, 1)
ylim(-1.2, 1.2)
scatter(train_x, train_y)
plot(x, y)
show()
