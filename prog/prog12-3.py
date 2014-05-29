# -*- coding: utf8 -*-
import random
import numpy as np
from matplotlib.pyplot import *
import matplotlib.cm as cmap

np.random.seed(1)

N = 40
train_x = np.zeros((N, 2))
train_t = np.zeros(N, dtype=int)
train_x[0:N/2,:] = np.random.normal([0.5, 0], [0.2,0.4], (N/2, 2))
train_x[N/2:N,:] = np.random.normal([-0.5, 0], [0.2,0.4], (N-N/2,2))
train_t[0:N/2] = 1
train_t[N/2:N] = -1

#=== SMO法によるSVMの最適化 ===
# SMO法の反復回数上限
MAX_ITER = 1000

# |x|<ZERO_EPS のとき, x=0が成立していると見なす.
ZERO_EPS = 1.0e-2

def kernel(x1, x2):
    return x1.dot(x2)

def gram_matrix(xs):
    K = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            K[i,j] = kernel(xs[i,:], xs[j,:])
    return K

# 識別関数
def discriminant(mu, theta, x):
    v = 0
    for i in range(N):
        if mu[i] < ZERO_EPS: continue
        v += mu[i]*train_t[i]*kernel(train_x[i,:], x)
    return v + theta

#=== SMO法 ===

# 定数項を計算
def threshold(mu, K):
    v = 0
    n = 0
    for i in range(N):
        if mu[i] < ZERO_EPS: continue
        n += 1
        s = train_t[i]
        for j in range(N):
            if mu[j] < ZERO_EPS: continue
            s -= mu[j]*train_t[j]*K[i, j]
        v += s
    return v/n

# KKT条件が成立しているならTrue
def checkKKT(i, mu, theta, K):
    yi = train_t[i]*discriminant(mu, theta, train_x[i,:])
    return yi >= 1 and mu[i]*(yi-1) < ZERO_EPS

# 2つ目の更新ベクトルを選択
def choose_second(i, mu):
    di = discriminant(mu, 0, train_x[i,:])
    m  = 0
    mj = 0
    for j in range(N):
        if mu[j] < ZERO_EPS: continue
        v = abs(discriminant(mu, 0, train_x[j,:])-di)
        if v > m:
            m = v
            mj = j
    return mj

def update_mu(mu, i, j, K):
    if i == j: return False
    ti       = train_t[i]
    tj       = train_t[j]
    di       = discriminant(mu, 0, train_x[i,:])
    dj       = discriminant(mu, 0, train_x[j,:])
    delta_i  = (1-ti*tj+ti*(dj-di))/(K[i,i]-2*K[i,j]+K[j,j])
    next_mui = mu[i] + delta_i
    c        = ti*mu[i] + tj*mu[j]
    if ti == tj:
        lim = c/ti
        if next_mui < 0: next_mui = 0.0
        elif next_mui > lim: next_mui = lim
    else:
        lim = max(c/ti, 0)
        if next_mui < lim : next_mui = lim
    if abs(next_mui - mu[i]) < ZERO_EPS: return False
    mu[i] = next_mui
    mu[j] = (c-ti*mu[i])/tj
    return True

#== 学習 ==
# 以下のメインループはアイデアを理解してもらう為に
# 非常に単純化したものです.
# もっと高速な方法はSMO法の論文を参照してください.

mu = np.random.uniform(0, 1, N)
K  = gram_matrix(train_x)
theta = threshold(mu, K)
for p in range(MAX_ITER):
    print mu
    changed = False
    for i in range(N):
        if checkKKT(i, mu, theta, K): continue
        j = choose_second(i, mu)
        changed = update_mu(mu, i, j, K) or changed
        theta = threshold(mu, K)
    if not changed: break
print "count=",p

# 表示
X, Y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
Z = np.vectorize(lambda x, y: discriminant(mu, theta, np.array([x,y])))(X, Y)
#for i in range(N):
#    alpha = 0.3
#    if mu[i] >= ZERO_EPS:
#        alpha = 1.0
#    plot(train_x[i,0], train_x[i,1], marker="o", color=["red", "blue"][(train_t[i]+1)/2], markersize=10, alpha=alpha)
#pcolor(X, Y, Z, alpha=0.3)
#show()
Z = Z <= 0

xlim(-1, 1)
ylim(-1, 1)
for i in range(N):
    alpha = 0.3
    if mu[i] >= ZERO_EPS:
        alpha = 1.0
    plot(train_x[i,0], train_x[i,1], marker="o", color=["red", "blue"][(train_t[i]+1)/2], markersize=10, alpha=alpha)
pcolor(X, Y, Z, alpha=0.3)
title("2-class SVM")
show()
