# -*- coding: utf8 -*-
import random
import numpy as np
from matplotlib.pyplot import *
import matplotlib.cm as cmap

np.random.seed(0)

N = 30
train_x = np.random.uniform(-1, 1, (N, 2))
train_t = 2*(train_x[:,0] < train_x[:,1])-1
print train_t
#train_t = 2*np.logical_and(train_x[:,0]<0, train_x[:,1]<0)-1

#=== SVM ===
# カーネル関数の分散パラメータ
KERN_SIGMA = 0.8

def kernel(x1, x2):
    return np.exp(-sum((x1-x2)**2)/(2*KERN_SIGMA**2))

def gram_matrix(xs):
    K = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            K[i,j] = kernel(xs[i,:], xs[j,:])
    return K

# 識別関数(ここではコードをシンプルにする為, 全ベクトルを使っている)
def discriminant(mu, theta, x):
    v = 0
    for i in range(N):
        v += mu[i]*train_t[i]*kernel(train_x[i], x)
    return v + theta

#=== SMO法 ===
# |x|<ZERO_EPS のとき, x=0が成立していると見なす.
ZERO_EPS = 1.0e-2

# 定数項を計算
def threshold(mu, K):
    v = 0
    n = 0
    for i in range(N):
        if mu[i] < ZERO_EPS: continue
        n += 1
        s = 0
        for j in range(N):
            if mu[j] < ZERO_EPS: continue
            s += train_t[i] - mu[j]*train_t[j]*K[i, j]
        v += s
    return v/n

# 定数項を近似的に計算
def approx_threshold(mu, K):
    return np.average(train_t - K.dot(mu*train_t))

# KKT条件が成立しているならTrue
def checkKKT(i, mu, theta, K):
    yi = train_t[i]*discriminant(mu, theta, train_x[i])
    return yi >= 1 and mu[i]*(yi-1) < ZERO_EPS

# 2つ目の更新ベクトルを選択
def choose_second(i, mu):
    di = discriminant(mu, 0, train_x[i])
    m  = 0
    mj = 0
    for j in range(N):
        #if mu[j] < ZERO_EPS: continue
        v = abs(discriminant(mu, 0, train_x[j])-di)
        if v > m:
            m = v
            mj = j
    return mj

def update_mu(mu, i, j, K):
    if i == j: return False
    ti       = train_t[i]
    tj       = train_t[j]
    di       = discriminant(mu, 0, train_x[i])
    dj       = discriminant(mu, 0, train_x[j])
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

mu = np.random.uniform(0, 1, N)
print mu
K  = gram_matrix(train_x)
theta = threshold(mu, K)
changed = True
while changed:
    changed = False
    for i in range(N):
        if checkKKT(i, mu, theta, K): continue
        changed = True
        j = choose_second(i, mu)
        update_mu(mu, i, j, K)
        print mu
        theta = threshold(mu, K)
theta = threshold(mu, K)
print mu

# 表示
X, Y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
Z = np.vectorize(lambda x, y: discriminant(mu, theta, np.array([x,y])))(X, Y)

scatter(train_x[:,0], train_x[:,1], c=train_t, s=50, cmap=cmap.cool)
pcolor(X, Y, Z, alpha=0.3)
for i in range(N):
    if mu[i] > ZERO_EPS:
        plot(train_x[i,0], train_x[i,1], "bo", markersize=10)
show()

Z = Z > 0.5
scatter(train_x[:,0], train_x[:,1], c=train_t, s=50, cmap=cmap.cool)
pcolor(X, Y, Z, alpha=0.3)
show()
