# -*- coding: utf-8 -*-
from numpy import *
from matplotlib.pyplot import *

#### サンプルデータ ####
N = [10, 10, 10]
mu = [array([0, 0]), array([1, 2]), array([3, 1])]
sigma = [0.5, 0.5, 0.5]

xs = []; ys = []; cs = []
for i in range(len(N)):
    for j in range(N[i]):
        xs.append(random.normal(mu[i][0], sigma[i]))
        ys.append(random.normal(mu[i][1], sigma[i]))
        cs.append(i)

scatter(xs, ys, c=cs, s=50, linewidth=0)
show()

savetxt("data1-8-1", [xs,ys,cs], delimiter="\t")
