# -*- coding: utf-8 -*-
from numpy import *
N = 1000

# N(0,1) に従う乱数を N 個生成
x = random.randn(N)

print average((x-0)**2)
