# -*- coding: utf-8 -*-
from numpy import *
from scipy import linalg as LA
from matplotlib.pyplot import *
import matplotlib.cm as cm

N = 30

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
    # 最小二乗法
    X = array([xs**k for k in range(d+1)]).T
    A = LA.solve(X.T.dot(X), X.T.dot(ys))

    # グラフ生成
    x = linspace(0, pi, 100)
    y = A.dot(array([x**k for k in range(d+1)]))

    # 残差平方和
    v = A.dot(array([xs**k for k in range(d+1)])) - ys
    RSS = v.T.dot(v)

    # AIC
    AIC = N*log(RSS/N)+2*d

    title("polynomial model (dim=%d), RSS=%.3f AIC=%.3f+C" % (d, RSS, AIC))
    xlabel('x')
    ylabel('y')
    xlim(0,pi)
    ylim(0,1)
    scatter(xs, ys)
    plot(x, y)
    savefig("fig3-10-%d.png" % d)

xlabel('x')
ylabel('y')
xlim(0,pi)
ylim(0,1)
scatter(xs, ys)
savefig("fig3-10-0.png")

for d in range(1,10):
    clf()
    fit(d)
