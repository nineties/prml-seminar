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
    y = random.normal(sin(x), 0.05)
    xs[i] = x
    ys[i] = y

def fit(d):
    # 最小二乗法
    X = array([xs**k for k in range(d+1)]).T
    A = LA.solve(X.T.dot(X), X.T.dot(ys))

    # グラフ生成
    x = linspace(0, pi, 100)
    y = A.dot(array([x**k for k in range(d+1)]))

    title("polynomial model (dim=%d)" % d)
    xlabel('x')
    ylabel('y')
    xlim(0,pi)
    ylim(0,1)
    scatter(xs, ys)
    plot(x, y)
    savefig("fig2-4-%d.png" % d)

xlabel('x')
ylabel('y')
xlim(0,pi)
ylim(0,1)
scatter(xs, ys)
savefig("fig2-4-0.png")

for d in range(1,10):
    clf()
    fit(d)
