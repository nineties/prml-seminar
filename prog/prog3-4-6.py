# -*- coding: utf-8 -*-
from numpy import *
from scipy import stats
from matplotlib.pyplot import *

x = linspace(1.5, 4, 1000)
y1 = stats.norm.pdf(x)
plot(x, y1)
fill_between(x[where(x>=2)], y1[where(x>=2)], 0, alpha=0.3)

plot(x, stats.gamma.pdf(x, a=2, loc=2)/10)
show()
