# -*- coding: utf-8 -*-
from numpy import *

P = array([[1.0/2, 1.0/3, 2.0/3],
           [    0, 1.0/3, 1.0/3],
           [1.0/2, 1.0/3,     0]])

p = array([1.0, 0.0, 0.0])

for t in range(20):
    print "t=%d\t:" % t,
    print p
    p = P.dot(p)
