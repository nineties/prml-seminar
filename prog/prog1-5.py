# -*- coding: utf-8 -*-
from numpy import *
from matplotlib.pyplot import *
from scipy.spatial import Voronoi, voronoi_plot_2d

# 図を作るためのコードなので代表ベクトルは決め打ちで作ってます
N = [10, 10, 10, 10]
mu = [array([0, 0]), array([2, 0]), array([0, 2]), array([1.5,1.5])]
cat = [0, 0, 0, 1]
sigma = [0.5, 0.5]
color = ['red', 'green']
xrng = [-2, 4]
yrng = [-2, 4]

xs = []; ys = []
xavg = []; yavg = []
for i in range(len(N)):
    x = []
    y = []
    p = mu[i]
    s = sigma[cat[i]]
    for j in range(N[i]):
        x.append(random.normal(p[0], s))
        y.append(random.normal(p[1], s))
    xs.append(x)
    ys.append(y)
    xavg.append(average(x))
    yavg.append(average(y))

def plot_points(repr):
    xlim(xrng)
    ylim(yrng)
    for i in range(len(N)):
        if repr:
            plot(xavg[i], yavg[i], "bo", color='yellow')
        scatter(xs[i], ys[i], s=60, c=color[cat[i]])

voronoi_plot_2d(Voronoi(array([xavg, yavg]).T))
plot_points(True)
savefig("fig1-5.png")

