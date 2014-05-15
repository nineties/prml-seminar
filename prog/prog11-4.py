# -*- coding: utf-8 -*-
from numpy import *
from scipy import linalg as LA
from matplotlib.pyplot import *

random.seed(0)

# Gaussian process regression
N = 10
train_x = random.uniform(0, 1, N)
train_t = random.normal(sin(2*pi*train_x), 0.1)

# 精度パラメータ
BETA = 2

# カーネル関数
THETA0 = 1
THETA1 = 4
THETA2 = 0
THETA3 = 5
def k(x1, x2):
    return THETA0 * exp(-THETA1*(x1-x2)**2/2) + THETA2 + THETA3*x1*x2

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
x = linspace(0, 1, 100)
y = vectorize(m)(x)

# 予測値とp(t|x)と標準偏差
X, Y = meshgrid(linspace(0, 1, 100), linspace(-2.0, 2.0, 100))
Z = vectorize(cond_p)(X, Y)
xlim(0, 1)
ylim(-2.0, 2.0)
scatter(train_x, train_t)
plot(x, y)
plot(x, y+vectorize(sigma)(x))
plot(x, y-vectorize(sigma)(x))
pcolor(X, Y, Z, alpha=0.3)
title(u"θ=(%.2f, %.2f, %.2f, %.2f)" % (THETA0, THETA1, THETA2, THETA3))
show()
