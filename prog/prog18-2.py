# -*- coding: utf-8 -*0
from numpy import *
from matplotlib.pyplot import *
import matplotlib.cm as cm
from sklearn.cluster import KMeans

N = 150
x = concatenate([
    random.multivariate_normal([0, 0], eye(2), N/3),
    random.multivariate_normal([0, 5], eye(2), N/3),
    random.multivariate_normal([2, 3], eye(2), N/3),
    ])

fig = figure(figsize=(16,6))
ax0 = fig.add_subplot( 121 )
ax1 = fig.add_subplot( 122 )

ax0.set_xlim(-3, 5)
ax0.set_ylim(-3, 8)
ax0.scatter(x[:, 0], x[:, 1], s=50, cmap=cm.cool)

clf = KMeans(n_clusters=3)
clf.fit(x)

X, Y = meshgrid(linspace(-3, 5), linspace(-3, 8))
Z = vectorize(lambda x,y: clf.predict([x, y]))(X, Y)
ax1.set_xlim(-3, 5)
ax1.set_ylim(-3, 8)
ax1.scatter(x[:, 0], x[:, 1], s=50, cmap=cm.cool)
ax1.pcolor(X, Y, Z, alpha=0.2)
savefig('fig18-2.png')
