# -*- coding: utf-8 -*-
from numpy import *
from scipy import linalg as LA
from matplotlib.pyplot import *
import matplotlib.cm as cm

xmin = 0; xmax=2
ymin = 0; ymax=1

# [0,1] x [0,1] 内のサンプルを生成
# 識別面は y = x

N = 30
xs = random.uniform(0, 1, N)
ys = random.uniform(0, 1, N)

def fit(xs, ys, N):

    # クラス割り当て
    cs = ys > xs

    # 目標ベクトル
    def unit(i):
        t = zeros(2)
        t[i] = 1
        return t

    ts = zeros((N, 2))
    for i in range(N):
        ts[i] = unit(cs[i])

    # 最小二乗法による学習
    # 計画行列
    X = array([xs, ys, ones(N)]).T
    w = LA.solve(X.T.dot(X), X.T.dot(ts))

    def classify(w, x, y):
        p = w.T.dot([x,y,1])
        if LA.norm(p-unit(0)) > LA.norm(p-unit(1)):
            return 0
        else:
            return 1

    X, Y = meshgrid(linspace(xmin, xmax, 100), linspace(ymin, ymax, 100))
    Z = vectorize(lambda x,y: classify(w,x,y))(X, Y)
    xlim(xmin, xmax)
    ylim(ymin, ymax)
    pcolor(X, Y, Z, alpha=0.3)
    scatter(xs, ys, c=cs, s=50, linewidth=1, cmap=cm.cool)
    show()

fit(xs, ys, N)
# 外れ値を一個だけ挿入
xs = append(xs, [1.9])
ys = append(ys, [0.1])
fit(xs, ys, N+1)
