from numpy import *
from matplotlib.pyplot import *

mu = 0
sigma = 1
x = linspace(-10, 10, 100)
plot(x, 1/(1+exp(-(x-mu)/sigma)))
show()
