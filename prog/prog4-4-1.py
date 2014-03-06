from numpy import *
from matplotlib.pyplot import *

N = 30

# y = 2x + 1 + e
a = 2
b = 1
sigma = 0.5

x = random.uniform(0, 1, N)
y = random.normal(2*x+1, sigma)

scatter(x, y)
show()

savetxt("prog4-4.dat", array([x,y]).T)
