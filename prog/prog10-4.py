# -*- coding: utf-8 -*-
from numpy import *
from scipy import linalg as LA
from matplotlib.pyplot import *

# Nadaraya-Watsonモデル
N = 10
train_x = random.uniform(0, 1, N)
train_y = random.normal(sin(2*pi*train_x), 0.1)

SIGMA = 0.1

# m(x) = E[p(y|x)]
def m(x):
    kernels = exp(-(x-train_x)**2/(2*SIGMA**2))
    return kernels.dot(train_y)/sum(kernels)

# 標準偏差
def std(x):
    kernels = exp(-(x-train_x)**2/(2*SIGMA**2))
    Ey  = kernels.dot(train_y)/sum(kernels)
    Ey2 = kernels.dot(train_y**2+SIGMA**2)/sum(kernels)
    return sqrt(Ey2-Ey**2)

def p(x, y):
    return average(exp(-((x-train_x)**2 + (y-train_y)**2)/(2*SIGMA**2))/(2*pi*SIGMA**2))

# p(y|x)
def cond_p(x, y):
    num   = p(x,y)
    denom = average(exp(-(x-train_x)**2/(2*SIGMA**2))/sqrt(2*pi*SIGMA**2))
    return num/denom

# 予測値
x = linspace(0, 1, 100)
y = vectorize(m)(x)

# p(x,y)
X, Y = meshgrid(linspace(0, 1, 100), linspace(-1.1, 1.1, 100))
Z = vectorize(p)(X, Y)

# 予測値のみ
xlim(0, 1)
ylim(-1.1, 1.1)
scatter(train_x, train_y)
plot(x, y)
show()

# 予測値とp(x,y)
xlim(0, 1)
ylim(-1.1, 1.1)
scatter(train_x, train_y)
plot(x, y)
pcolor(X, Y, Z, alpha=0.3)
show()

# 予測値とp(y|x)
Z = vectorize(cond_p)(X, Y)
xlim(0, 1)
ylim(-1.1, 1.1)
scatter(train_x, train_y)
plot(x, y)
pcolor(X, Y, Z, alpha=0.3)
show()

# 予測値と標準偏差
std = vectorize(std)(x)
xlim(0, 1)
ylim(-1.1, 1.1)
scatter(train_x, train_y)
plot(x, y)
plot(x, y+std)
plot(x, y-std)
show()
