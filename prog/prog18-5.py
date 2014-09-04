# -*- coding: utf-8 -*0
from numpy import *
from matplotlib.pyplot import *
import scipy.stats as stats

x = linspace(-3, 5, 100)
dist1 = stats.norm(0, 1)
dist2 = stats.norm(3, 0.5)
pi1 = 1.0/3
pi2 = 2.0/3

plot(x, dist1.pdf(x), linestyle='dotted')
plot(x, dist2.pdf(x), linestyle='dotted')
plot(x, pi1*dist1.pdf(x) + pi2*dist2.pdf(x))
savefig('prog18-5.png')
