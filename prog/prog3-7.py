# -*- coding: utf-8 -*-
from numpy import *
from matplotlib.pyplot import *
from collections import Counter

P = array([[1.0/2, 1.0/3, 2.0/3],
           [    0, 1.0/3, 1.0/3],
           [1.0/2, 1.0/3,     0]])

def next(x):
    u = random.uniform()
    p = 0.0
    for i in range(3):
        p += P[i,x]
        if u <= p: return i

BURNIN = 10
N = 1000

# バーンイン
x = 0
for i in range(BURNIN): x = next(x)

samples = zeros(N, dtype=int)
for i in range(N):
    samples[i] = x
    x = next(x)

hist,_ = histogram(samples, bins=3)
print hist/float(N)
