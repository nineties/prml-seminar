from numpy import *
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D

N = 200
x = random.uniform(0, 10, N)
y = random.uniform(0, 10, N)
z = x + 2*y + 1 + random.normal(0, 2, N)

ax = Axes3D(figure())
ax.scatter(x, y, z, color='red')

X, Y = meshgrid(linspace(0, 10, 10), linspace(0, 10, 10))
Z = X + 2*Y + 1
ax.plot_wireframe(X, Y, Z)
show()
