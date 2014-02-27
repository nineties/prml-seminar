# -*- coding: utf-8 -*-
from numpy import *
from matplotlib.pyplot import *

M = 100
def error(n):
    x = random.randn(n)
    return abs(1 - average((x-0)**2))

def average_error(n):
    return average([error(n) for i in range(M)])

xs = arange(100, 10001, 100)
ys = vectorize(average_error)(xs)
plot(xs, ys, label="MAE")
plot(xs, 1/sqrt(xs), label="1/sqrt(N)")

xlabel('number of samples (N)')
ylabel('mean absolute error (MAE)')
legend(loc=1)
show()
