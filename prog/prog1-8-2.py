# -*- coding: utf-8 -*-
import sys
from numpy import *
from scipy import linalg
from matplotlib.pyplot import *

data = loadtxt(sys.argv[1], delimiter="\t")

xs = data[0:2,:].T   # 学習データ [[x0,y0], [x1,y1], ...]
cs = int_(data[2,:]) # 正解クラス [c0, c1, ...]

# 学習データの数
N = len(xs)

# クラスの数
K = max(cs)+1

#### 特徴ベクトルの変換関数 ####
def hi(x):

#### パラメータの最適化 ####
# phi(特徴ベクトル)を行に並べた行列
X = array([ phi(xs[i]) for i in range(N) ])

# 正解クラス番号の成分だけ1にしたベクトルpを
# 行に並べた行列
P = zeros([N, K])
for i in range(N):
    P[i, cs[i]] = 1

# X^TXA = X^TP を満たす A が求めるパラメータ
A = linalg.solve(X.T.dot(X), X.T.dot(P))

# 擬似逆行列を使う場合
# A = linalg.pinv(X).dot(P)

# 最小二乗法を実行してくれる関数もあります
# A,residues,rank,s=linalg.lstsq(X, P)

#### 識別器の構築 ####
def distance(x, i):
    t = zeros(K)
    t[i] = 1
    return linalg.norm(x-t)

# (x, y) と (0,...,0,1,0,...,0) の距離が最小のクラスに分類
def classify(x, y):
    p = A.T.dot(phi([x, y]))
    return argmin([distance(p, i) for i in range(K)])

#### どんな感じで空間が分割されたか見てみましょう ####
# 表示領域の設定
xmin = min(xs[:,0]); xmax = max(xs[:,0])
ymin = min(xs[:,1]); ymax = max(xs[:,1])
xmin -= (xmax-xmin)/20
xmax += (xmax-xmin)/20
ymin -= (ymax-ymin)/20
ymax += (ymax-ymin)/20
xlim(xmin, xmax)
ylim(ymin, ymax)

X, Y = meshgrid(linspace(xmin, xmax, 100), linspace(ymin, ymax, 100))
Z = vectorize(classify)(X, Y)
pcolor(X, Y, Z, alpha=0.1)
scatter(xs[:,0], xs[:,1], c=cs, s = 50, linewidth=0)
show()
