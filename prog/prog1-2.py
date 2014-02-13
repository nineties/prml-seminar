from numpy import *
from matplotlib.pyplot import *
from scipy.spatial import Voronoi, voronoi_plot_2d

representative = [array([0, 0]), array([1, 2]), array([3, 1]), array([3, 3])]
color = ['red', 'green', 'blue', 'yellow']

def plot_points():
    xlim([-1, 4])
    ylim([-1, 4])

    for i in range(len(representative)):
        p = representative[i]
        plot(p[0], p[1], "bo", color=color[i])
        text(p[0]+0.1, p[1]+0.1, "class%d" % i)

plot_points()
savefig("fig1-2-1.png")

voronoi_plot_2d(Voronoi(representative))
plot_points()
savefig("fig1-2-2.png")

