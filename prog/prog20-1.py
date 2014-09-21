# -*- coding: utf-8 -*-
from numpy import *
from scipy.special import psi
from scipy.misc import logsumexp
import pandas as pd
from sklearn.preprocessing import LabelEncoder

random.seed(1)

zoo = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data', header=None)
attributes = [u'体毛', u'羽毛', u'卵', u'乳', u'飛ぶ', u'水棲', u'肉食', u'歯', u'脊椎', u'呼吸', u'毒', u'ひれ', u'足', u'尾', u'家畜化', u'猫サイズ']

zoo.columns = [u'名前'] + attributes + [u'種別']

# 足の数を連番に変換
legs_enc = LabelEncoder()
zoo[u'足'] = legs_enc.fit_transform(zoo[u'足'])

# 学習に用いるデータのセットアップ
x = zoo[attributes].as_matrix()
N, M = x.shape
nj = x.max(0) + 1

# クラス数
K = 3

# 事前分布のパラメータ
alpha = random.uniform(1.0, 1.1, K)
beta  = empty(M, dtype=object)
for j in range(M):
    beta[j] = ones(nj[j])

# 学習
A = tile(alpha, (N, 1))
B = beta.repeat(K).reshape(-1, K)
r = zeros((N,M,K))

for it in range(1000):
    psi_Aik  = psi(A)
    psi_Ai   = psi(A.sum(1))
    psi_Bjk  = zeros((M, K))
    psi_Bjkl = empty((M, K), dtype=object)
    for j in xrange(M):
        for k in xrange(K):
            psi_Bjk[j,k]  = psi(B[j,k].sum())
            psi_Bjkl[j,k] = psi(B[j,k])

    new_r = zeros((N, M, K))
    for i in xrange(N):
        for j in xrange(M):
            for k in xrange(K):
                new_r[i, j, k] = psi_Aik[i,k]-psi_Ai[i]+psi_Bjkl[j,k][x[i,j]]-psi_Bjk[j, k]
    new_r -= logsumexp(new_r, axis=2).reshape(N, M, 1)
    new_r = exp(new_r)

    # rが変化しなくなったら終了
    if linalg.norm(new_r - r) < 1e-3*linalg.norm(r):
        break
    r = new_r

    A = alpha + r.sum(1)
    for j in xrange(M):
        for k in xrange(K):
            B[j,k] = beta[j].copy()
            for i in xrange(N):
                B[j,k][x[i,j]] += r[i,j,k]
print 'count =',it+1

# 結果の出力
#== クラス-属性値分布(MAP推定) ==
print '==== per class attribute distributions ===='
for k in range(K):
    print '> class %d' % k
    for j in range(M):
        print '%s:' % attributes[j],
        for l in range(nj[j]):
            print '\t[%d] %.3f' % (l, (B[j, k][l]-1)/(B[j, k].sum()-nj[j])),
        print ''


print '<table>'
print '<tr><th></th>',
for j in range(M):
    print '<th>%s</th>' % attributes[j],
print '</tr>'

for k in range(K):
    print u'<tr><th>クラス%d</th>' % k,
    for j in range(M):
        p = (B[j, k]-1)/(B[j, k].sum()-nj[j])
        color='black'
        if p.max() > 0.99: # 99%を超えている属性は色を付ける 
            color='red'
        if attributes[j] != u'足':
            print '<td style="color:%s">%s</td>' % (color, ['N', 'Y'][p.argmax()]),
        else:
            print '<td style="color:%s">%d本</td>' % (color, legs_enc.classes_[p.argmax()]),
    print '</tr>'
print '</table>'

#== 生物-所属クラス ==
print '==== per animal class distributions ===='
for i in range(N):
    print zoo[u'名前'][i],':',
    for k in range(K):
        print '\t[%d] %.3f' % (k, (A[i, k] - 1)/(A[i].sum() - K)),
    print ''

#== 属性-所属クラス ==
print u'名前',
for j in range(M):
    print '\t%s' % attributes[j],
print ''
for i in range(N):
    print zoo[u'名前'][i],
    for j in range(M):
        print '\t%d' % r[i, j].argmax(),
    print ''
