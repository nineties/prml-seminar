# -*- coding: utf-8 -*-
from numpy import *
from scipy import linalg as LA
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D

N = 100

#=== 学習データ ===
D = 5
K = 4

#平面 x = s*u + t*v 上に各クラスの中心を配置してみる.
u = array([1,2,1,3,2])
v = array([2,-3,0,1,1])

center = [u+v, u-v, -u+v, -u-v]
sigma  = [2,2,2,2]

train_x = zeros((N, D))
train_t = zeros(N)
for i in range(N):
    c = random.randint(K)
    train_x[i] = random.normal(center[c], sigma[c], D)
    train_t[i] = c

output = zeros((N, D+1))
output[:,0:D] = train_x
output[:,D] = train_t
savetxt("prog6-5.dat", output)

#=== DD次元に削減 ==
DD = 2

# 各クラスの標本数
Ns = [count_nonzero(train_t == i) for i in range(K)]

# 全標本の平均
mu = average(train_x, axis=0)

# 各クラスの平均
mus = [average(train_x[where(train_t == i)], axis=0) for i in range(K)]

# クラス内共分散とクラス間共分散
def iprod(x): return x.T.dot(x)

Sw = zeros((D, D))
for i in range(K):
    Sw += iprod(train_x[where(train_t == i)] - mus[i])

Sb = zeros((D, D))
for i in range(K):
    Sb += Ns[i] * outer(mus[i]-mu, mus[i]-mu)

# 固有値・固有ベクトルを求める
lam, vec = LA.eig(LA.inv(Sw).dot(Sb))

# 固有ベクトルは固有値の大きい順番に並んでいるので, 最初からDD個取って変換行列を作る.
W = float_(vec[:,0:DD])

# 実際に変換をしてプロットしてみる.
y = train_x.dot(W)
scatter(y[:,0], y[:,1], c=train_t, s=50, cmap=cm.gist_rainbow)
title("Fisher's linear discriminant")
savefig("fig6-5.png")
