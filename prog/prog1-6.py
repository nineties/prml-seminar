from numpy import *
from matplotlib.pyplot import *

N = [5, 5, 5]
mu = [array([0, 0]), array([1, 3]), array([4, 1])]
sigma = [0.8, 0.8, 0.8]
color = ['red', 'green', 'blue']
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

xlim(xrng)
ylim(yrng)
for i in range(len(N)):
    text(average(xs[i]), average(ys[i]), "class%d" % i)
    scatter(xs[i], ys[i], s=60, c=color[i])

# input
x = 2
y = 0
distance = []
for i in range(len(N)):
    for j in range(N[i]):
        distance.append( (i, j, (xs[i][j]-x)**2 + (ys[i][j]-y)**2) )

plot(x, y, "bo", color="black")
text(x+0.1, y+0.1, "x")
distance = sorted(distance, key=lambda x: x[2])
for k in range(5):
    i,j,d = distance[k]
    arrow(x, y, xs[i][j]-x, ys[i][j]-y, color=color[i], width=0.01)

savefig("fig1-6.png")
