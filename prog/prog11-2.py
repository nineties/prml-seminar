# -*- coding: utf8 -*-
from numpy import *
from matplotlib.pyplot import *

# ガウス過程
# mu(x) = 0
# k(x, x') = exp(-|x-x'|^2/(2sigma^2))
# からのサンプリング

N = 200

# カーネル関数
def k(x1, x2):
    THETA = 1
    return exp(-THETA*abs(x1-x2))

# データ点
x = linspace(-1, 1, N)

# グラム行列
K = zeros((N, N))
for i in range(N):
    for j in range(N):
        K[i, j] = k(x[i], x[j])

xlim(-1, 1)
ylim(-3, 3)
for i in range(5):
    y = random.multivariate_normal(zeros(N), K)
    plot(x, y)
show()
