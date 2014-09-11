# -*- coding: utf-8 -*-
from sklearn.datasets import fetch_mldata
from matplotlib.pyplot import *
from numpy import *
# MNISTの手書き数字データベース
# (初回実行時にダウンロードが行われます.)
mnist = fetch_mldata('MNIST original')
data  = array(mnist.data != 0, dtype=bool) # 二値化
gray()
matshow(data[0].reshape(28, 28))
savefig('fig19-1.png')
