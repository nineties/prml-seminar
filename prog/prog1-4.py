from numpy import *
from matplotlib.pyplot import *
from scipy.spatial import Voronoi, voronoi_plot_2d

N = [10, 10, 20]
mu = [array([0, 0]), array([1, 3]), array([4, 1])]
sigma = [0.3, 0.3, 2.0]
color = ['red', 'green', 'blue']
xrng = [-2, 5]
yrng = [-2, 4]

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
    xlim(xrng)
    ylim(yrng)
    for i in range(len(N)):
        if repr:
            text(xavg[i]+0.1, yavg[i]+0.1, "representative of class%d" % i)
            plot(xavg[i], yavg[i], "bo", color='yellow')
        scatter(xs[i], ys[i], s=60, c=color[i])

voronoi_plot_2d(Voronoi(array([xavg, yavg]).T))
plot_points(True)
savefig("fig1-4-1.png")

clf()

plot_points(True)
X, Y = meshgrid(arange(xrng[0], xrng[1], 0.1), arange(yrng[0], yrng[1], 0.1))
diff = [((X-xavg[i])**2 + (Y-yavg[i])**2)/sigma[i]**2 for i in range(len(N))]

contour(X, Y, (diff[0]-diff[1]), [0])
contour(X, Y, (diff[0]-diff[2]), [0])
contour(X, Y, (diff[1]-diff[2]), [0])
savefig("fig1-4-2.png")
