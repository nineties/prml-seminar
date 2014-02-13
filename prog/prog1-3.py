from numpy import *
from matplotlib.pyplot import *
from scipy.spatial import Voronoi, voronoi_plot_2d

N = [10, 10, 10]
mu = [array([0, 0]), array([1, 2]), array([3, 1])]
sigma = [0.5, 0.5, 0.5]
color = ['red', 'green', 'blue']

xs = []; ys = []
xavg = []; yavg = []
for i in range(len(N)):
    x = []
    y = []
    p = mu[i]
    s = sigma[i]
    for j in range(N[i]):
        x.append(random.normal(p[0], s))
        y.append(random.normal(p[1], s))
    xs.append(x)
    ys.append(y)
    xavg.append(average(x))
    yavg.append(average(y))

def plot_points(repr):
    xlim([-2, 5])
    ylim([-2, 4])
    for i in range(len(N)):
        if repr:
            text(xavg[i]+0.1, yavg[i]+0.1, "representative of class%d" % i)
            plot(xavg[i], yavg[i], "bo", color='yellow')
        scatter(xs[i], ys[i], s=60, c=color[i])

plot_points(False)
savefig("fig1-3-1.png")

plot_points(True)
savefig("fig1-3-2.png")

voronoi_plot_2d(Voronoi(array([xavg, yavg]).T))
plot_points(True)
savefig("fig1-3-3.png")
