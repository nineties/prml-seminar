# -*- coding: utf-8 -*-
from sklearn.datasets import fetch_mldata
from matplotlib.pyplot import *
from numpy import *

mnist = fetch_mldata('MNIST original')
data  = array(mnist.data != 0, dtype=bool) # 二値化

# 適当に15サンプル表示
N = len(data)
choice = random.choice(arange(N), 15)
figure(figsize=(18,8))
gray()
for i in range(15):
    subplot(3, 5, i+1)
    imshow(data[choice[i]].reshape(28, 28), interpolation='none')
savefig('fig19-2.png')
