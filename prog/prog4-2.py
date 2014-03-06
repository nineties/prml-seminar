# -*- coding: utf-8 -*-
from numpy import *
from scipy import linalg as LA
from math import gamma
from matplotlib.pyplot import *
import matplotlib.cm as cm

# ディリクレ分布
a = [3,2,5]
z = sum([gamma(x) for x in a])/gamma(sum(a))
def p(x):
    if x[0] + x[1] > 1:
        return -0.1
    return x[0]**a[0] * x[1]**a[1] * (1-x[0]-x[1])**a[2] / z

X, Y = meshgrid(linspace(0, 1, 100), linspace(0, 1, 100))
Z = vectorize(lambda x,y: p([x,y]))(X, Y)

xlim(0, 1)
ylim(0, 1)
pcolor(X, Y, Z, alpha=0.3)
title("Dirichlet distribution a=%s" % a)
show()

# Gibbs sampling
def next(x):
    new_x = random.beta(a[0]+1, a[2]+1)*(1-x[1])
    new_y = random.beta(a[1]+1, a[2]+1)*(1-new_x)
    return [new_x, new_y]

BURNIN = 10 # グラフの見やすさの為に小さな値にしています
N = 100

x = [0, 0]

# バーンイン
burn_x = zeros(BURNIN); burn_y = zeros(BURNIN)
for i in range(BURNIN):
    burn_x[i] = x[0]
    burn_y[i] = x[1]
    x = next(x)

# サンプリング
sample_x = zeros(N); sample_y = zeros(N)

# グラフの見た目を良くするために１点共有
sample_x[0] = burn_x[-1]
sample_y[0] = burn_y[-1]

for i in range(1, N):
    sample_x[i] = x[0]
    sample_y[i] = x[1]
    x = next(x)

xlim(0, 1)
ylim(0, 1)
pcolor(X, Y, Z, alpha=0.3)
plot(burn_x, burn_y, label="burn-in")
plot(sample_x, sample_y, label="sampling")
title("Dirichlet distribution a=%s" % a)
legend(loc=1)
show()
