# -*- coding: utf-8 -*-
from numpy import *
from scipy import stats
from matplotlib.pyplot import *

for l in arange(0.5, 3, 0.5):
    N = 100
    x = linspace(0, 3, N)
    plot(x, stats.expon.pdf(x, scale=1/l), label="lambda=%.1f" % l)
legend(loc=1)
show()
