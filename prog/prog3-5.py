# -*- coding: utf-8 -*-
from numpy import *
from scipy import stats
from matplotlib.pyplot import *

# 棄却サンプリングのパラメータ
M = 2

# 目標とする分布
def f(x):
    return 1.0-abs(x)

x = linspace(-1, 1, 100)
y = vectorize(f)(x)
xlim(-1, 1)
ylim(0, 2)

plot(x, y, label="f(x)")
legend(loc=1)
show()

plot(x, y, label="f(x)")
plot(x, stats.beta.pdf(x, 2, 2, loc=-1, scale=2)*M, label="beta(2,2) * %.1f" % M)
legend(loc=1)
show()

#棄却サンプリングで分布fの分散を求めてみる.
ngen = 0
def sample():
    global ngen
    while True:
        ngen += 1
        x = 2*random.beta(2, 2)-1
        p = f(x)/(M*stats.beta.pdf(x, 2, 2, loc=-1, scale=2))
        if random.uniform() < p: return x

N = 1000
samples = array([sample() for i in range(N)])
plot(x, y, label="f(x)")
hist(samples, bins=50, normed=True, alpha=0.3)
show()

print "total number of samples = %d" % ngen
print "adoption rate = %.2f" % (float(N)/ngen)
