# -*- coding: utf-8 -*-
from numpy import *
from scipy import linalg as LA
from matplotlib.pyplot import *

# カーネル法による多項式フィッティング

# 学習用データ
N = 10
train_x = random.uniform(0, 1, N)
train_y = random.normal(sin(2*pi*train_x), 0.1)

# 多項式モデルの次数
M = 3000

# 正則化項の係数
LAMBDA = 0.001

# カーネル関数
def k(x1, x2):
    v = 0
    for i in range(M+1):
        v += x1**i * x2**i
    return v

# グラム行列
K = zeros((N, N))
for i in range(N):
    for j in range(N):
        K[i,j] = k(train_x[i], train_x[j])

# 学習 (ここでは K + LAMBDA*I の正則性チェックは省略)
a = LA.solve(K + LAMBDA*identity(N), train_y)

# 学習結果の表示
x = linspace(0, 1, 100)
y = zeros(100)
for i in range(100):
    y[i] = a.dot([k(train_x[j], x[i]) for j in range(N)])

xlim(0, 1)
ylim(-1.2, 1.2)
scatter(train_x, train_y)
plot(x, y)
show()
