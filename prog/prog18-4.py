# -*- coding: utf-8 -*-
from numpy import *
from skimage import io
from sklearn.cluster import KMeans

K = 2
pixels = io.imread('../fig/lego.jpeg')
original_shape = pixels.shape

# 浮動小数点数配列に変換してK-means
x = pixels.reshape(-1, 3).astype(float)
clf = KMeans(n_clusters = K)
clf.fit(x)
cls = clf.predict(x)

# 各クラスの所属ピクセルを重心の値に置き換える.
for k in range(K):
    x[cls == k] = x[cls == k].mean(0)

pixels = x.astype(uint8).reshape(*original_shape)
io.imsave('fig18-4-%d.png' % K, pixels)
