# -*- coding: utf-8 -*-
from numpy import *
from matplotlib.pyplot import *

#### サンプルデータ ####

# 学習データの数
N = 50

# 単位円 x^2+y^2=1 の内と外の2クラスにしてみます
xs = []; ys = []; cs = []
for i in range(N):
    x = random.uniform(-1.5, 1.5)
    y = random.uniform(-1.5, 1.5)
    xs.append(x); ys.append(y)
    if x**2 + y**2 < 1:
        cs.append(0)
    else:
        cs.append(1)

scatter(xs, ys, c=cs, s=50, linewidth=0)
show()

savetxt("data1-8-3", [xs,ys,cs], delimiter="\t")
