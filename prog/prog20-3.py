# -*- coding: utf-8 -*-
from numpy import *
from scipy.special import psi
from scipy.misc import logsumexp
import pandas as pd
from sklearn.preprocessing import LabelEncoder

random.seed(1)

zoo = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data', header=None)
attributes = [u'体毛', u'羽毛', u'卵', u'乳', u'飛ぶ', u'水棲', u'肉食', u'歯', u'背骨', u'呼吸', u'毒', u'ひれ', u'足', u'尾', u'家畜化', u'猫サイズ']

zoo.columns = [u'名前'] + attributes + [u'種別']

# 足の数を連番に変換
legs_enc = LabelEncoder()
zoo[u'足'] = legs_enc.fit_transform(zoo[u'足'])

# 学習に用いるデータのセットアップ
x = zoo[attributes].as_matrix()
N, M = x.shape
nj = x.max(0) + 1

# クラス数
K = 10

# 事前分布のパラメータ
alpha = concatenate([
    random.uniform(3.0, 3.1, 3),
    random.uniform(1.0, 1.1, K-3)
    ])
beta  = empty(M, dtype=object)
for j in range(M):
    beta[j] = ones(nj[j])

# 周辺化ギブスサンプリング
BURNIN  = 100
NSAMPLE = 1000

# 各生物 i の属性 j の所属クラスをランダムに初期化
z = zeros((N, M), dtype=int)
for i in range(N):
    for j in range(M):
        z[i, j] = random.randint(nj[j])
beta_Njkl = zeros((M, K), dtype=object)
for j in range(M):
    for k in range(K):
        beta_Njkl[j, k] = beta[j].copy()
        for i in range(N):
            if z[i, j] != k:
                continue
            beta_Njkl[j, k][x[i, j]] += 1

alpha_Mik = zeros((N, K))
for i in range(N):
    for k in range(K):
        alpha_Mik[i, k] = alpha[k]
        for j in range(M):
            alpha_Mik[i, z[i, j]] += 1

z_hist = zeros((N, M, K), dtype=int)
for it in range(BURNIN + NSAMPLE):
    if it%10==0:
        print it
        print z_hist
    for i in xrange(N):
        for j in xrange(M):
            # x[i,j], z[i,j] の頻度情報だけデクリメント
            alpha_Mik[i, z[i,j]] -= 1
            beta_Njkl[j, z[i, j]][x[i, j]] -= 1

            # サンプリング分布の確率を計算
            p = zeros(K)
            for k in range(K):
                p[k] = alpha_Mik[i, k] * beta_Njkl[j, k][x[i, j]] / beta_Njkl[j, k].sum()
            p /= p.sum()

            new_zij = random.choice(K, p = p)

            z[i, j] = new_zij

            if it >= BURNIN:
                z_hist[i, j, new_zij] += 1

            alpha_Mik[i, z[i,j]] += 1
            beta_Njkl[j, z[i, j]][x[i, j]] += 1

print z_hist

#== 属性-所属クラス ==
print u'名前',
for j in range(M):
    print '\t%s' % attributes[j],
print ''
for i in range(N):
    print zoo[u'名前'][i],
    for j in range(M):
        print '\t%d' % z_hist[i, j].argmax(),
    print ''


