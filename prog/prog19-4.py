# -*- coding: utf-8 -*-
from sklearn.datasets import fetch_20newsgroups
from scipy.misc import logsumexp, factorial
from scipy.sparse import csr_matrix
from scipy.special import gammaln
import gensim
from matplotlib.pyplot import *
from numpy import *
import shelve

df = fetch_20newsgroups()

N = len(df.data)
print u'文書数:',N

# ステミング等の前処理
data = []
for i in xrange(N):
    data.append( gensim.parsing.preprocess_string(df.data[i]) )

# 単語の辞書を作成し，極端な語を除去
dictoinary = gensim.corpora.Dictionary(data)
dictoinary.filter_extremes()

print u'単語数:',len(dictoinary)
M = len(dictoinary)

# Bag of Words表現に変換
# ついでに L と log L!/(x1! ... xM!) も計算しておく.
rows = []
cols = []
vals = []
C = zeros(N)
L = zeros(N)
for i in xrange(N):
    l = 0
    for j, v in dictoinary.doc2bow(data[i]):
        rows.append(i)
        cols.append(j)
        vals.append(v)

        C[i] -= gammaln(v+1)
        l += v
    C[i] += gammaln(l+1)
    L[i] = l

x = csr_matrix( (vals, (rows, cols)), shape=(N, M), dtype=float )
xT = x.T

# 学習
K = 10
print u'学習するクラス数:',K

p  = random.uniform(0.25, 0.75, (K, M))
p /= c_[p.sum(1)]
pi = ones(K)/K
r  = zeros( (N, K) )
for it in range(10000):
    # E step
    r[:] = log(pi) + x.dot(ma.log(p.T)) + c_[C]
    r -= c_[logsumexp(r, axis=1)]
    r = exp(r)

    # M step
    new_p  = xT.dot(r).T / c_[L.dot(r)]
    new_pi = r.sum(0)/N

    if linalg.norm(new_p - p) < 0.1*linalg.norm(p): break

    p  = new_p
    pi = new_pi

print 'count =', it+1

p_sorted = argsort(p, axis=1)
for k in range(K):
    idxs = set(p_sorted[k, -50:])
    for j in range(K):
        if k == j: continue
        idxs -= set(p_sorted[j, -50:])
    print 'topic %d:' % k,
    for w in map(lambda i: dictoinary[i], idxs):
        print '%s,' % w,
    print ''
