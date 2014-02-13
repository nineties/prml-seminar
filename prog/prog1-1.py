from numpy import *
from matplotlib.pyplot import *

N = [10, 10, 10]
mu = [array([0, 0]), array([1, 2]), array([3, 1])]
sigma = [0.5, 0.5, 0.5]
color = ['red', 'green', 'blue']

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

for i in range(len(N)):
    text(average(xs[i]), average(ys[i]), "class%d" % i)
    scatter(xs[i], ys[i], s=60, c=color[i])

savefig("fig1-1.png")
show()
