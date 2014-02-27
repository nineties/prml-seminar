from numpy import *
from scipy import stats
from matplotlib.pyplot import *
x = linspace(1.5, 4, 1000)
y = stats.norm.pdf(x)
plot(x, y)
fill_between(x[where(x>=2)], y[where(x>=2)], 0, alpha=0.3)
show()
