# -*- coding: utf-8 -*-
from numpy import *
from scipy import linalg as LA
from matplotlib.pyplot import *
import matplotlib.cm as cm

# 多変量分布の図示
mu = array([0, 0]) # 平均
S  = array([[1, 0.5],[0.5, 1]]) # 分散
Sinv = LA.inv(S)
detS = LA.det(S)

def f(x):
    return exp(-(x-mu).T.dot(Sinv).dot(x-mu)/2)/(2*pi*sqrt(detS))

X, Y = meshgrid(linspace(-3, 3, 100), linspace(-3, 3, 100))
Z = vectorize(lambda x,y: f([x,y]))(X, Y)

# MH法
sigma=1
def next(x):
    while True:
        new_x = x + random.normal(0, sigma, 2)
        if random.uniform() <= min( f(new_x)/f(x), 1 ):
            return new_x

BURNIN = 100 # グラフの見やすさの為に小さな値にしています
N = 1000

x = [1, -1]

# バーンイン
burn_x = zeros(BURNIN); burn_y = zeros(BURNIN)
for i in range(BURNIN):
    burn_x[i] = x[0]
    burn_y[i] = x[1]
    x = next(x)

# サンプリング
sample_x = zeros(N); sample_y = zeros(N)
for i in range(N):
    sample_x[i] = x[0]
    sample_y[i] = x[1]
    x = next(x)

xlim(-3, 3)
ylim(-3, 3)
pcolor(X, Y, Z, alpha=0.3)
scatter(burn_x, burn_y, label="burn-in", color="blue")
scatter(sample_x, sample_y, label="sampling", color="red")
legend(loc=1)
title("sigma=%.1f, number of samples=%d, burn-in=%d" % (sigma, N, BURNIN))
show()
