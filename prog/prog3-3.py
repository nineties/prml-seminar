# -*- coding: utf-8 -*-
from numpy import *
N = 10000

# [-1,1] x [-1,1] 上の一様乱数を N 点生成
x = random.uniform(-1, 1, N)
y = random.uniform(-1, 1, N)

# 円内に入った点の比率から円周率を計算
print 4.0*count_nonzero(x**2 + y**2 <= 1)/N
