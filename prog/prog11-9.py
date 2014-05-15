# -*- coding: utf-8 -*-
from numpy import *
from scipy import stats
from matplotlib.pyplot import *

a = linspace(-10, 10, 100)
y1 = 1/(1+exp(-a))
y2 = stats.norm.cdf(sqrt(pi/8)*a, 0, 1)
plot(a, y1, label=u"σ(a)")
plot(a, y2, label=u"Φ(λa)")
legend(loc=1)
show()
