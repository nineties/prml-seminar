from numpy import *
from matplotlib.pyplot import *

N = [10, 10]
mu = [array([0, 0]), array([1, 3])]
sigma = [1.0, 1.0]
color = ['red', 'green']
xrng = [-2, 5]
yrng = [-2, 4]

xs = []; ys = []
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

# noise
xs[1][0] = 1
ys[1][0] = 0

xlim(xrng)
ylim(yrng)
for i in range(len(N)):
    text(average(xs[i]), average(ys[i]), "class%d" % i)
    scatter(xs[i], ys[i], s=60, c=color[i])

# input
x = 1.5
y = 0
distance = []
for i in range(len(N)):
    for j in range(N[i]):
        distance.append( (i, j, (xs[i][j]-x)**2 + (ys[i][j]-y)**2) )

plot(x, y, "bo", color="black")
text(x+0.1, y+0.1, "x")
distance = sorted(distance, key=lambda x: x[2])
for k in range(1):
    i,j,d = distance[k]
    arrow(x, y, xs[i][j]-x, ys[i][j]-y, color=color[i], width=0.01)

savefig("fig1-7.png")
