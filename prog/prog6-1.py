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

BURNIN = 1000
RANDOM_SIGMA = 1 # ランダム・ウォークの標準偏差パラメータ

NSAMPLES = 10000

#=== 事前分布 ===
w0 = zeros(DIM)
S0inv = LA.inv( 100*identity(DIM) )

#=== 学習データ ===
# クラス1: y<5-x^2 or (x-5)^2+(y-5)^2<2
train_x = random.uniform(xmin, xmax, N)
train_y = random.uniform(ymin, ymax, N)
train_t = logical_or(train_y < 5 - (train_x-2)**2, (train_x-8)**2+(train_y-7)**2<3)

xlim(xmin, xmax)
ylim(ymin, ymax)
scatter(train_x, train_y, c=train_t, s=50, cmap=cm.cool)
title("Training data")
savefig("fig6-1-training.png")

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
savefig("fig6-1-bases.png")

#=== シグモイド関数 ===
def sigmoid(x):
    return 1/(1+exp(-x))

#=== 計画行列 ===
X = array([phi(train_x[i], train_y[i]) for i in range(N)])

#=== p(C_1|x)のプロット ===
def show_iteration(i, w):
    clf()
    X, Y = meshgrid(linspace(xmin, xmax, 100), linspace(ymin, ymax, 100))
    Z = vectorize(lambda x,y: sigmoid(w.dot(phi(x,y))))(X, Y)
    xlim(xmin, xmax)
    ylim(ymin, ymax)
    pcolor(X, Y, Z, alpha=0.3)
    scatter(train_x, train_y, c=train_t, s=50, cmap=cm.cool)
    title("MCMC (iteration=%d)" % i)
    savefig("fig6-1-iter%d.png" % i)

#==== MCMC ====
def accept_ratio(w, new_w):
    s = sigmoid(X.dot(w))
    new_s = sigmoid(X.dot(new_w))
    num = sum(train_t*log(new_s) + (1-train_t)*log(1-new_s)) \
          -0.5*(new_w-w0).T.dot(S0inv).dot(new_w-w0)
    denom = sum(train_t*log(s) + (1-train_t)*log(1-s)) \
            -0.5*(w-w0).T.dot(S0inv).dot(w-w0)
    return exp(num - denom)

def sampler(w0):
    w = w0
    for i in range(BURNIN):
        new_w = w + random.normal(0, RANDOM_SIGMA**2, DIM)
        if random.uniform() < accept_ratio(w, new_w):
            w = new_w
    while True:
        new_w = w + random.normal(0, RANDOM_SIGMA**2, DIM)
        if random.uniform() < accept_ratio(w, new_w):
            w = new_w
            yield w

show_iteration(0, w0)
sim = sampler(w0)
for i in range(NSAMPLES):
    w = sim.next()
    if (i+1)%1000 == 0:
        show_iteration(i+1, w)
show_iteration(NSAMPLES, w)
