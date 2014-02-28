# -*- coding: utf-8 -*-
from numpy import *
from scipy import linalg as LA
from matplotlib.pyplot import *
import matplotlib.cm as cm

N = 10

# サンプルデータ
# y=sin(x) (0 < x < pi) を平均値に.
xs = zeros(N)
ys = zeros(N)
for i in range(N):
    x = random.uniform(0, pi)
    y = random.normal(sin(x), 0.1)
    xs[i] = x
    ys[i] = y

def fit(d):
    title("polynomial model (dim=%d)" % d)
    xlabel('x')
    ylabel('y')
    xlim(0,pi)
    ylim(0,1)
    scatter(xs, ys)
    # 単純な最小二乗法
    X = array([xs**k for k in range(d+1)]).T
    a = LA.solve(X.T.dot(X), X.T.dot(ys))

    # グラフ生成
    x = linspace(0, pi, 100)
    y = a.dot(array([x**k for k in range(d+1)]))
    plot(x, y, label="linear regression")

    # パラメータの事前分布の平均・分散の逆行列
    a0 = zeros(d+1)
    s0inv = LA.inv(diag([2**(-k) for k in range(d+1)]))
    # 誤差分布の分散. 本当は何らかの推定で決めますが, とりあえず1に．
    s = 1
    # ベイズ線形回帰
    a = LA.solve(X.T.dot(X)/s**2 + s0inv, X.T.dot(ys)/s**2 + s0inv.dot(a0))
    x = linspace(0, pi, 100)
    y = a.dot(array([x**k for k in range(d+1)]))
    plot(x, y, label="bayesian linear regression")
    legend(loc=3)
    savefig("fig3-13-%d.png" % d)

xlabel('x')
ylabel('y')
xlim(0,pi)
ylim(0,1)
scatter(xs, ys)
savefig("fig3-13-0.png")

for d in range(1,10):
    clf()
    fit(d)
