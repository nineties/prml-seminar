from numpy import *
from matplotlib.pyplot import *

mu = 0
sigma = 1
x = linspace(-10, 10, 100)
plot(x, exp(-(x-mu)**2/(2*sigma**2)))
show()
