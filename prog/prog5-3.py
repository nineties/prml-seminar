# -*- coding: utf-8 -*-
from numpy import *
from scipy import linalg as LA
from matplotlib.pyplot import *
import matplotlib.cm as cm

N = 100
xmin = 0; xmax = 10
ymin = 0; ymax = 10

BASIS_SIGMA = 3.0
BASIS_COUNT = 4
DIM = BASIS_COUNT*BASIS_COUNT # 特徴空間の次元

NUM_ITERATIONS = 100 # 反復回数

ETA=1.0 # 学習率はETA/(利用したデータ数)で調整

#=== 学習データ ===
# クラス1: y<5-x^2 or (x-5)^2+(y-5)^2<2
train_x = random.uniform(xmin, xmax, N)
train_y = random.uniform(ymin, ymax, N)
train_t = logical_or(train_y < 5 - (train_x-2)**2, (train_x-8)**2+(train_y-7)**2<3)

xlim(xmin, xmax)
ylim(ymin, ymax)
scatter(train_x, train_y, c=train_t, s=50, cmap=cm.cool)
title("Training data")
savefig("fig5-3-training.png")

#=== 基底 ===
# ガウス基底を BASIS_COUNT^2 個等間隔で配置することにする
basis_center = [(x, y) for x in linspace(xmin, xmax, BASIS_COUNT)
                       for y in linspace(ymin, ymax, BASIS_COUNT)]

# 中心(cx, cy) のガウス基底の点(x,y)での値
def gaussian_basis(x, y, cx, cy):
    return exp(-((x-cx)**2 + (y-cy)**2)/(2*BASIS_SIGMA**2))

# 基底関数ベクトルへの変換関数
def phi(x,y):
    return array([gaussian_basis(x, y, cx, cy) for (cx,cy) in basis_center])

# 基底を表示(半径=標準偏差 の円)
xlim(xmin, xmax)
ylim(ymin, ymax)
for cx,cy in basis_center:
    gcf().gca().add_artist(Circle((cx, cy), BASIS_SIGMA, fill=False))
title("Placement of gaussian bases")
savefig("fig5-3-bases.png")

#=== シグモイド関数 ===
def sigmoid(x):
    return 1/(1+exp(-x))

#=== p(C_1|x)のプロット ===
def show_iteration(i, w):
    clf()
    X, Y = meshgrid(linspace(xmin, xmax, 100), linspace(ymin, ymax, 100))
    Z = vectorize(lambda x,y: sigmoid(w.dot(phi(x,y))))(X, Y)
    xlim(xmin, xmax)
    ylim(ymin, ymax)
    pcolor(X, Y, Z, alpha=0.3)
    scatter(train_x, train_y, c=train_t, s=50, cmap=cm.cool)
    title(u"stochastic gradient descent (number of data=%d)" % i)
    savefig("fig5-3-iter%d.png" % i)

#==== 確率的勾配降下法 ====
w = zeros(DIM)
eta = 10.0
show_iteration(0, w)
for i in range(2*N):
    if i%10 == 0:
        show_iteration(i, w)
    x = train_x[i%N]
    y = train_y[i%N]
    t = train_t[i%N]
    grad = (t - sigmoid(phi(x,y).dot(w)))* phi(x, y)
    w += eta/(i+1) * grad
show_iteration(2*N, w)
