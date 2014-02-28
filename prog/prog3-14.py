# -*- coding: utf-8 -*-
from numpy import *
from scipy import linalg as LA
from matplotlib.pyplot import *
import matplotlib.cm as cm

N = 30
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
    title("polynomial model (dim=%d)" % d)
    xlabel('x')
    ylabel('y')
    xlim(0,pi)
    ylim(0,1)
    scatter(xs, ys, color='red', label="training")
    scatter(txs, tys, color='blue', label="validation")

    # 最小二乗法
    X = array([xs**k for k in range(d+1)]).T
    a = LA.solve(X.T.dot(X), X.T.dot(ys))

    # グラフ生成
    x = linspace(0, pi, 100)
    y = a.dot(array([x**k for k in range(d+1)]))
    # 残差平方和
    v = a.dot(array([txs**k for k in range(d+1)])) - tys
    RSS = v.T.dot(v)
    plot(x, y, label="non bayes, RSS=%.3f" % RSS)

    ## ベイズ線形回帰
    # パラメータの事前分布の平均・分散の逆行列
    a0 = zeros(d+1)
    s0inv = LA.inv(diag([2**(-k) for k in range(d+1)]))
    # 誤差分布の分散. 本当は何らかの推定で決めますが, とりあえず1に．
    s = 1
    a = LA.solve(X.T.dot(X)/s**2 + s0inv, X.T.dot(ys)/s**2 + s0inv.dot(a0))

    # グラフ生成
    x = linspace(0, pi, 100)
    y = a.dot(array([x**k for k in range(d+1)]))
    # 残差平方和
    v = a.dot(array([txs**k for k in range(d+1)])) - tys
    RSS = v.T.dot(v)
    plot(x, y, label="bayes, RSS=%.3f" % RSS)

    legend(loc=3)
    savefig("fig3-14-%d.png" % d)

xlabel('x')
ylabel('y')
xlim(0,pi)
ylim(0,1)
scatter(xs, ys, color='red', label="training")
scatter(txs, tys, color='blue', label="validation")
legend(loc='upper right')
savefig("fig3-14-0.png")

for d in range(1,10):
    clf()
    fit(d)
