from numpy import *
from matplotlib.pyplot import *

beta = 0.1
x = linspace(0, 1, 100)
for mu in linspace(0, 1, 10):
    plot(x, exp(-(x-mu)**2/(2*beta**2)))
title("Gaussian basis (beta=%.1f)" % beta)
show()
