# -*- coding: utf-8 -*-
from numpy import *
from scipy import linalg as LA
from matplotlib.pyplot import *
import matplotlib.cm as cm

N = 20
K = 4 # 分割数
group_size = N//K
colors = ['red', 'blue', 'green', 'yellow']

# サンプルデータ
# y=sin(x) (0 < x < pi) を平均値に.
xs = []
ys = []
for i in range(N):
    x = random.uniform(0, pi)
    y = random.normal(sin(x), 0.05)
    xs.append(x)
    ys.append(y)

def fit_one_group(i, d):
    # 検証用
    x = array(xs[i*group_size:(i+1)*group_size])
    y = array(ys[i*group_size:(i+1)*group_size])

    # 訓練用
    tx = copy(xs).tolist()
    ty = copy(ys).tolist()
    del tx[i*group_size:(i+1)*group_size]
    del ty[i*group_size:(i+1)*group_size]
    tx = array(tx)
    ty = array(ty)

    # 最小二乗法
    X = array([tx**k for k in range(d+1)]).T
    A = LA.solve(X.T.dot(X), X.T.dot(ty))

    # 残差平方和
    v = A.dot(array([x**k for k in range(d+1)])) - y

    # グラフ生成
    gx = linspace(0, pi, 100)
    gy = A.dot(array([gx**k for k in range(d+1)]))

    RSS = v.T.dot(v)
    scatter(x, y, color=colors[i])
    plot(gx, gy, color=colors[i])
    return RSS

def fit(d):
    xlabel('x')
    ylabel('y')
    xlim(0,pi)
    ylim(0,1)

    averageRSS = average([fit_one_group(i, d) for i in range(K)])
    print "d=%d: averageRSS=%.3f" % (d, averageRSS)

    title("polynomial model (dim=%d) average RSS=%.3f" % (d, averageRSS))
    legend(loc='upper right')
    savefig("fig2-6-%d.png" % d)

for d in range(1,10):
    clf()
    fit(d)
