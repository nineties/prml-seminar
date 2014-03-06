# -*- coding: utf-8 -*-
from numpy import *
from scipy import stats
from matplotlib.pyplot import *

x = linspace(1e-10, 3, 100)
params = [[1,1],[2,1],[1,2],[2,2]]
for a,b in params:
    plot(x, stats.invgamma.pdf(x, a, scale=b), label=u"α=%.1f, β=%.1f"%(a,b))
title("Inverse gamma distribution")
legend(loc=1)
show()
