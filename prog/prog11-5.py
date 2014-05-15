# -*- coding: utf8 -*-
from numpy import *
from matplotlib.pyplot import *

# ガウス過程
# mu(x) = 0
# k(x, x') = exp(-|x-x'|^2/(2sigma^2))
# からのサンプリング

N = 200

# パラメータ
ALPHA = 0.01

# カーネル関数
def k(x1, x2):
    SIGMA = 0.3
    return exp(-(x1-x2)**2/(2*SIGMA**2))

# データ点
x = linspace(-1, 1, N)

# グラム行列
K = zeros((N, N))
for i in range(N):
    for j in range(N):
        K[i, j] = k(x[i], x[j])

a = random.multivariate_normal(zeros(N), K/ALPHA)
y = 1/(1 + exp(-a))

xlim(-1, 1)
ylim(-10, 10)
plot(x, a)
show()

xlim(-1, 1)
ylim(0, 1)
plot(x, y)
show()
