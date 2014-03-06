# -*- coding: utf-8 -*-
from numpy import *
from scipy import linalg as LA
from matplotlib.pyplot import *
import matplotlib.cm as cm

xmin = 0; xmax=2
ymin = 0; ymax=2

# [0,2] x [0,2] 内のサンプルを生成
# 識別面は y = x-1, y=x+1

N = 50
xs = random.uniform(xmin, xmax, N)
ys = random.uniform(ymin, ymax, N)

def fit(xs, ys, N):
    K = 3

    # クラス割り当て
    cs = zeros(N)
    cs[where(ys > xs + 0.5)] = 1
    cs[where(ys < xs - 0.5)] = 2

    # 目標ベクトル
    def unit(i):
        t = zeros(K)
        t[i] = 1
        return t

    ts = zeros((N, K))
    for i in range(N):
        ts[i] = unit(cs[i])

    # 最小二乗法による学習
    # 計画行列
    X = array([xs, ys, ones(N)]).T
    w = LA.solve(X.T.dot(X), X.T.dot(ts))

    def classify(w, x, y):
        p = w.T.dot([x,y,1])
        return argmax([LA.norm(p-unit(i)) for i in range(K)])

    X, Y = meshgrid(linspace(xmin, xmax, 100), linspace(ymin, ymax, 100))
    Z = vectorize(lambda x,y: classify(w,x,y))(X, Y)
    xlim(xmin, xmax)
    ylim(ymin, ymax)
    pcolor(X, Y, Z, alpha=0.3, cmap=cm.cool)
    scatter(xs, ys, c=cs, s=50, linewidth=1, cmap=cm.cool)
    show()

fit(xs, ys, N)
