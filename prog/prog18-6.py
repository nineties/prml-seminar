# -*- coding: utf-8 -*0
from numpy import *
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats

X, Y = meshgrid(linspace(-3, 9), linspace(-3, 20))

dist1 = stats.multivariate_normal([0,0], eye(2))
dist2 = stats.multivariate_normal([3,3], 2*eye(2))
dist3 = stats.multivariate_normal([2,8], 3*eye(2))

Z = vectorize(lambda x,y: dist1.pdf([x,y])/3 + dist2.pdf([x,y])/3 + dist3.pdf([x,y])/3)(X, Y)

ax = Axes3D(figure())
ax.plot_surface(X, Y, Z, rstride=1, cstride=1)
show()
