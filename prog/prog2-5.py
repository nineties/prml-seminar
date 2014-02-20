# -*- coding: utf-8 -*-
from numpy import *
from scipy import linalg as LA
from matplotlib.pyplot import *
import matplotlib.cm as cm

N = 20
T = N*2//3 # 2/3を検証用にする

# サンプルデータ
# y=sin(x) (0 < x < pi) を平均値に.
xs = []
ys = []
for i in range(N):
    x = random.uniform(0, pi)
    y = random.normal(sin(x), 0.05)
    xs.append(x)
    ys.append(y)

idxs = arange(N)
random.shuffle(idxs)

# 検証用
txs = array([xs[idxs[i]] for i in range(N-T,N)])
tys = array([ys[idxs[i]] for i in range(N-T,N)])

# 学習用
xs = array([xs[idxs[i]] for i in range(N-T)])
ys = array([ys[idxs[i]] for i in range(N-T)])

def fit(d):
    # 最小二乗法
    X = array([xs**k for k in range(d+1)]).T
    A = LA.solve(X.T.dot(X), X.T.dot(ys))

    # グラフ生成
    x = linspace(0, pi, 100)
    y = A.dot(array([x**k for k in range(d+1)]))

    # 残差平方和
    v = A.dot(array([txs**k for k in range(d+1)])) - tys
    RSS = v.T.dot(v)
    print "d=%d: RSS=%.3f" % (d, RSS)

    title("polynomial model (dim=%d) RSS=%.3f" % (d, RSS))
    xlabel('x')
    ylabel('y')
    xlim(0,pi)
    ylim(0,1)
    scatter(xs, ys, color='red', label="training")
    scatter(txs, tys, color='blue', label="validation")
    legend(loc='upper right')
    plot(x, y)
    savefig("fig2-5-%d.png" % d)

xlabel('x')
ylabel('y')
xlim(0,pi)
ylim(0,1)
scatter(xs, ys, color='red', label="training")
scatter(txs, tys, color='blue', label="validation")
legend(loc='upper right')
savefig("fig2-5-0.png")

for d in range(1,10):
    clf()
    fit(d)
