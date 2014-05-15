# -*- coding: utf-8 -*-
from numpy import *
from scipy import linalg as LA
from matplotlib.pyplot import *

random.seed(0)

#== 超パラメータの学習 ==
MAX_ITER = 2000
ITER_EPS = 1.0e-5

# 精度パラメータ
BETA = 1.0

# 事前分布のパラメータ
GA_K = array([2.0, 2.0, 2.0, 2.0])
GA_S = array([2.0, 2.0, 2.0, 2.0])

# Gaussian process regression
N = 10
train_x = random.uniform(0, 1, N)
train_t = random.normal(sin(2*pi*train_x), 0.1)

# 超パラメータ
# w = (theta0, theta1, theta2, theta3)

# カーネル関数
def k(w, x1, x2):
    return w[0] * exp(-w[1]*(x1-x2)**2/2) + w[2] + w[3]*x1*x2

# diffCi == diff(C, w[i])
def diffC0(w):
    K = zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = exp(-w[1]*(train_x[i]-train_x[j])**2/2)
    return K
def diffC1(w):
    K = zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = -w[0]*(train_x[i]-train_x[j])**2/2*\
                    exp(-w[1]*(train_x[i]-train_x[j])**2/2)
    return K
def diffC2(w):
    return ones((N, N))
def diffC3(w):
    K = zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = train_x[i]*train_x[j]
    return K

# K + w[0]I の逆行列
def compute_Cinv(w):
    # グラム行列
    K = zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = k(w, train_x[i], train_x[j])
    return LA.inv(K + identity(N)/BETA)

# 対数尤度の勾配 x -1
def diff(w):
    Cinv   = compute_Cinv(w)
    Ct     = Cinv.dot(train_t)
    diffCs = [diffC0(w), diffC1(w), diffC2(w), diffC3(w)]
    d      = zeros(4)
    for i in range(4):
        d[i] = (-trace(Cinv.dot(diffCs[i])) + Ct.T.dot(diffCs[i]).dot(Ct))/2
    return d - 1/GA_S+ (GA_K-1)/log(w)

def steepest():
    w = random.uniform(0, 10, 4)
    for i in range(MAX_ITER):
        dw = 0.1*diff(w)
        if LA.norm(dw) < ITER_EPS: break
        w += dw
    print i
    return w

w = steepest()
print w

# 学習結果を使ってC^-1を作り直す
Cinv = compute_Cinv(w)
a = Cinv.dot(train_t)

# 予測値
def m(w, x):
    y = 0
    for i in range(N):
        y += a[i] * k(w, train_x[i], x)
    return y

# 標準偏差
def sigma(w, x):
    vk    = array([k(w, train_x[i], x) for i in range(N)])
    return sqrt(k(w, x, x) + 1.0/w[0]- vk.T.dot(Cinv).dot(vk))

# p(t|x)
def cond_p(w, x, t):
    mu = m(w, x)
    s  = sigma(w, x)
    return 1/sqrt(2*pi*s**2)*exp(-(t-mu)**2/(2*s**2))

# 予測値
x = linspace(0, 1, 100)
y = vectorize(lambda x: m(w, x))(x)

# 予測値とp(t|x)と標準偏差
X, Y = meshgrid(linspace(0, 1, 100), linspace(-2.0, 2.0, 100))
Z = vectorize(lambda x,y: cond_p(w, x, y))(X, Y)
xlim(0, 1)
ylim(-2.0, 2.0)
scatter(train_x, train_t)
plot(x, y)
plot(x, y+vectorize(lambda x:sigma(w, x))(x))
plot(x, y-vectorize(lambda x:sigma(w, x))(x))
pcolor(X, Y, Z, alpha=0.3)
title(u"θ=(%.2f, %.2f, %.2f, %.2f)" % (w[0], w[1], w[2], w[3]))
show()
