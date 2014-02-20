# -*- coding: utf-8 -*-
from numpy import *
from math import exp,lgamma
from sys import argv
from matplotlib.pyplot import *

def beta(x,y):
    return exp(lgamma(x)+lgamma(y)-lgamma(x+y))

x = linspace(0, 1, 100)
y = x**(5-1) * (1-x)**(2-1) / beta(5,2)
plot(x, y)

x = linspace(0, 1, 100)
y = x**(2-1) * (1-x)**(2-1) / beta(2,2)
plot(x, y)

xlabel(u'θ')
ylabel(u'π(θ)')

show()
