# -*- coding: utf-8 -*-
from numpy import *
from scipy import linalg as LA
from matplotlib.pyplot import *

# 第10回資料の図
N = 5
sample_x = random.uniform(-1, 1, N)

# 正規分布の場合
xlim(-1.5, 1.5)
ylim(-0.01, 1.2)
plot([-1.5, 1.5], [0, 0], color="black")
for i in range(N):
    plot(sample_x[i], 0, "bo")
    x = linspace(-1.5, 1.5, 100)
    y = 1/sqrt(2*pi*0.5**2)*exp(-(x-sample_x[i])**2/(2*0.5**2))
    plot(x, y, linestyle="dashed")

x = linspace(-1.5, 1.5, 100)
y = average([1/sqrt(2*pi*0.5**2)*exp(-(x-sample_x[i])**2/(2*0.5**2)) for i in range(N)], axis=0)
plot(x, y)
show()

# 窓関数

xlim(-1.5, 1.5)
ylim(-0.01, 1.2)
plot([-1.5, 1.5], [0, 0], color="black")
for i in range(N):
    plot(sample_x[i], 0, "bo")
    x = linspace(-1.5, 1.5, 100)
    y = abs(x-sample_x[i])<=0.5
    plot(x, y, linestyle="dashed")

x = linspace(-1.5, 1.5, 100)
y = average([abs(x-sample_x[i])<=0.5 for i in range(N)], axis=0)
plot(x, y)
show()
