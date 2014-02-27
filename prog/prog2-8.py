# -*- coding: utf-8 -*-
from numpy import *
from matplotlib.pyplot import *
import matplotlib.cm as cm

# クラス0: y>x, クラス1: y <= x としてサンプルデータ生成

N = 30
xs = []; ys = []; cs = []
for i in range(N):
    x = random.uniform(0, 1)
    y = random.uniform(0, 1)
    if y > x:
        c = 0
    else:
        c = 1
    xs.append(x)
    ys.append(y)
    cs.append(c)

# 1-NN法
def classify(x, y):
    i = argmin([(xs[i]-x)**2 + (ys[i]-y)**2 for i in range(N)])
    return cs[i]

X, Y = meshgrid(linspace(0, 1, 100), linspace(0, 1, 100))
Z = vectorize(classify)(X, Y)

xlim(0, 1)
ylim(0, 1)
scatter(xs, ys, c=cs, s=50, linewidth=1, cmap=cm.cool)
show()

xlim(0, 1)
ylim(0, 1)
pcolor(X, Y, Z, alpha=0.3)
scatter(xs, ys, c=cs, s=50, linewidth=1, cmap=cm.cool)
show()
