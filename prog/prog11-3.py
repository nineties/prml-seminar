# -*- coding: utf-8 -*-
from numpy import *
from scipy import linalg as LA
from matplotlib.pyplot import *

# Gaussian process regression
N = 500
train_x = random.uniform(0, 5, N)
train_t = random.normal(sin(pi*train_x) + 0.3*sin(3*pi*train_x), 0.1)

# 精度パラメータ
BETA = 100

# カーネル関数
def k(x1, x2):
    SIGMA = 0.05
    return sum([sin(i*pi*x1)*sin(i*pi*x2) for i in range(30)])
    # return sin(10*x1)*(sin(10*x2))
    #return exp(-(x1-x2)**2/(2*SIGMA**2))

# グラム行列
K = zeros((N, N))
for i in range(N):
    for j in range(N):
        K[i, j] = k(train_x[i], train_x[j])
Cinv = LA.inv(K + identity(N)/BETA)
a = Cinv.dot(train_t)

# 予測値
def m(x):
    y = 0
    for i in range(N):
        y += a[i] * k(train_x[i], x)
    return y

# 標準偏差
def sigma(x):
    vk    = array([k(train_x[i], x) for i in range(N)])
    return sqrt(k(x, x) + 1.0/BETA - vk.T.dot(Cinv).dot(vk))

# p(t|x)
def cond_p(x, t):
    mu = m(x)
    s  = sigma(x)
    return 1/sqrt(2*pi*s**2)*exp(-(t-mu)**2/(2*s**2))

# 予測値
x = linspace(0, 10, 100)
y = vectorize(m)(x)

# 予測値のみ
xlim(0, 10)
ylim(-2.0, 2.0)
scatter(train_x, train_t)
plot(x, y)
show()

# 予測値とp(t|x)と標準偏差
X, Y = meshgrid(linspace(0, 2, 100), linspace(-2.0, 2.0, 100))
Z = vectorize(cond_p)(X, Y)
xlim(0, 10)
ylim(-2.0, 2.0)
scatter(train_x, train_t)
plot(x, y)
plot(x, y+vectorize(sigma)(x))
plot(x, y-vectorize(sigma)(x))
pcolor(X, Y, Z, alpha=0.3)
show()
