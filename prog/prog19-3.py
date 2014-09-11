# -*- coding: utf-8 -*-
from sklearn.datasets import fetch_mldata
from matplotlib.pyplot import *
from numpy import *
from scipy.misc import logsumexp

mnist = fetch_mldata('MNIST original')
x  = array(mnist.data != 0, dtype=bool) # 二値化

# サンプルが多すぎるので適当に減らす
#N = 5000
#select = arange(len(x))
#random.shuffle(select)
#x = x[select[:N]]
N = len(x)

W = 28
H = 28

K  = 12
p  = random.uniform(0.25, 0.75, (K, W*H)) # 各分布のパラメータ
p  = p/c_[p.sum(1)]
pi = ones(K)/K
r  = zeros( (N, K) )
for it in range(1000):
    print it
    # E step
    r[:] = log(pi) + x.dot(ma.log(p.T)) + (~x).dot(ma.log(1-p.T))
    r -= c_[logsumexp(r, axis=1)]
    r = exp(r)

    # M step
    d = r.sum(0)
    new_p  = r.T.dot(x)/c_[d]
    new_pi = d/N

    if linalg.norm(new_p - p)/linalg.norm(p) < 1.0e-3: break

    p  = new_p
    pi = new_pi
print 'count =', it+1

# 学習された各クラスの平均
figure(figsize=(18,10))
gray()
for k in range(K):
    subplot(K/4, 4, k+1)
    title('pi=%.3f' % pi[k])
    imshow(p[k].reshape(H,W), interpolation='none')
savefig('fig19-3.png')

# 各クラスに所属する画像例
figure(figsize=(16,3))
for k in range(K):
    clf()
    idxs = argsort(r[:, k])[-5:]
    subplot(1, 5, 1)
    imshow(p[k].reshape(H, W), interpolation='none')
    for j in range(min(4, len(idxs))):
        subplot(1, 5, j+2)
        imshow(x[idxs[j]].reshape(H, W), interpolation='none')
    savefig('fig19-3-%d.png' % k)
