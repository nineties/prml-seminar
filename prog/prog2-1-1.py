# -*- coding: utf-8 -*-
from numpy import *
from math import exp,lgamma
from sys import argv
from matplotlib.pyplot import *

def beta(x,y):
    return exp(lgamma(x)+lgamma(y)-lgamma(x+y))

p = 2
q = 2

x = linspace(0, 1, 100)
y = x**(p-1) * (1-x)**(q-1) / beta(p,q)

xlabel(u'θ')
ylabel(u'π(θ)')

plot(x, y)
show()
