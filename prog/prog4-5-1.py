from numpy import *
from matplotlib.pyplot import *

N = 25
x = random.uniform(0, 1, N)
y = random.normal(sin(2*pi*x), 0.2)
xlim(0, 1)
scatter(x, y)
t = linspace(0, 1, 100)
plot(t, sin(2*pi*t))
show()

savetxt("prog4-5.dat", array([x,y]).T)
