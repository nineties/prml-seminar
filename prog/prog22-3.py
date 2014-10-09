# -*- coding: utf-8 -*-
from sklearn.datasets import fetch_mldata
from matplotlib.pyplot import *
from numpy import *

mnist = fetch_mldata('MNIST original')
data  = array(mnist.data != 0, dtype=bool) # 二値化

# 3だけ取る
x = data[mnist.target == 3].T

# 共分散行列
S = cov(x.T)

# 固有値・固有ベクトル
print linalg.eig(S)
