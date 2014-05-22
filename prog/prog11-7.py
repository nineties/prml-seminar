# -*- coding: utf8 -*-
from numpy import *
from matplotlib.pyplot import *
from scipy import linalg as LA
import matplotlib.cm as cmap

random.seed(0)

SIGMA = 0.4
NU = 0.1
ITER_MAX = 100

# 学習データ
N = 100
train_x = zeros((N, 2))
train_x[:, 0] = random.uniform(-1, 1, N)
train_x[:, 1] = random.uniform(-1, 1, N)

train_t = (train_x[:,1] - train_x[:,0])*(train_x[:,0]**2+train_x[:,1]**2-0.7)>0

# シグモイド
def sigmoid(a):
    return 1/(1+exp(-a))

def W(a):
    return diag(sigmoid(a)*(1-sigmoid(a)))

# カーネル関数
def kernel(x1, x2):
    return exp(-sum((x1-x2)**2)/(2*SIGMA**2))

# グラム行列
K = zeros((N, N))
for i in range(N):
    for j in range(N):
        K[i, j] = kernel(train_x[i,:], train_x[j,:])
C = K + identity(N)*NU

# aを求める
a = random.normal(0, 1, N)
for i in range(ITER_MAX):
    a = C.dot(LA.solve(identity(N) + W(a).dot(C), train_t-sigmoid(a)+W(a).dot(a)))


# y=p(t(n+1)|t)
def compute_y(a, x):
    k = zeros(N)
    for i in range(N):
        k[i] = kernel(train_x[i], x)
    mu = k.dot(train_t - sigmoid(a))
    sigma2 = kernel(x, x) + NU - k.dot(LA.inv(W(a)+C)).dot(k)
    return sigmoid(mu/sqrt(1+pi*sigma2/8))

X, Y = meshgrid(linspace(-1, 1, 100), linspace(-1, 1, 100))
Z = vectorize(lambda x, y: compute_y(a, array([x, y])))(X, Y)
print Z

# p(t(n+1)|t) の分布
xlim(-1, 1)
ylim(-1, 1)
pcolor(X, Y, Z, alpha=0.3)
scatter(train_x[:, 0], train_x[:, 1], c = train_t, s=50, cmap=cmap.cool)
title(u"σ=%.2f, ν=%.2f" % (SIGMA, NU))
show()

# 識別境界
Z[Z >  0.5] = 1
Z[Z <= 0.5] = 0
xlim(-1, 1)
ylim(-1, 1)
pcolor(X, Y, Z, alpha=0.3)
scatter(train_x[:, 0], train_x[:, 1], c = train_t, s=50, cmap=cmap.cool)
title(u"σ=%.2f, ν=%.2f" % (SIGMA, NU))
show()
