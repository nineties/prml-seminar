# -*- coding: utf-8 -*-
from numpy import *
from scipy import linalg as LA
from matplotlib.pyplot import *
import matplotlib.cm as cm

# 多変量分布の図示
mu = array([0, 0]) # 平均
S  = array([[1, 0.5],[0.5, 2]]) # 分散
Sinv = LA.inv(S)
detS = LA.det(S)

def N(x, y):
    v = array([x,y])
    return exp(-(v-mu).T.dot(Sinv).dot(v-mu)/2)/(2*pi*sqrt(detS))

X, Y = meshgrid(linspace(-3, 3, 100), linspace(-3, 3, 100))
Z = vectorize(N)(X, Y)

xlim(-3, 3)
ylim(-3, 3)

pcolor(X, Y, Z, alpha=0.3)
show()
