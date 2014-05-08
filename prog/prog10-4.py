# -*- coding: utf-8 -*-
from numpy import *
from scipy import linalg as LA
from matplotlib.pyplot import *

# Nadaraya-Watsonモデル
N = 10
train_x = random.uniform(0, 1, N)
train_y = random.normal(sin(2*pi*train_x), 0.1)

SIGMA = 0.1

def m(x):
    kernels = exp(-(x-train_x)**2/(2*SIGMA**2))
    return kernels.dot(train_y)/sum(kernels)

x = linspace(0, 1, 100)
y = vectorize(m)(x)

xlim(0, 1)
ylim(-1.1, 1.1)
scatter(train_x, train_y)
plot(x, y)
show()
