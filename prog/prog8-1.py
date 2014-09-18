# -*- coding: utf-8 -*-
from numpy import *
from scipy import linalg as LA
from matplotlib.pyplot import *

N = 50

D = 1   # 入力の次元
M = 3   # 隠れ層の数
K = 1   # 出力層の数

DIFF_EPS = 1.0e-2 # 中心差分法の摂動の大きさ

#=== 重みパラメータ ===
# 隠れ層の重みは M*(D+1) 行列w1で表現.
# 出力層の重みは K*(M+1) 行列w2で表現.
#
# w1[i, j] はj番目の入力と隠れ層のi番目の素子の間の重み.
# w2[i, j] は隠れ層のj番目素子と出力層のi番目の素子の間の重み.

#=== 順伝播 ===
# 入力 x に対して各素子への入力を計算する.
# 出力はタプル (a1, a2) であり,
# a1[i]が隠れ層の素子 i への入力, a2[i]が出力層の素子 i への入力

def forward(x, w1, w2):
    a1 = w1.dot(append(x, 1))         # 隠れ層への入力
    a2 = w2.dot(append(tanh(a1), 1))  # 出力層への入力
    return (a1, a2)

#=== 誤差逆伝播法 ===
# 各重みに関する偏微分係数を計算
def backpropagation(x, t, w1, w2):
    a1, a2 = forward(x, w1, w2)  # 順伝播
    delta2 = a2 - t              # 出力の誤差
    tanh_a1 = tanh(a1)
    delta1 = ((1-tanh_a1**2)*w2[:,0:M]).T.dot(delta2) # 隠れ層の誤差

    ## 偏微分係数の計算
    diff1 = zeros((M, D+1))
    diff2 = zeros((K, M+1))

    # 隠れ層
    diff1 = outer(delta1, append(x, 1))
    # 出力層
    diff2 = outer(delta2, append(tanh_a1, 1))
    return (diff1, diff2)

#=== 中心差分法 ===
#誤差関数
def E(x, t, w1, w2):
    a1, a2 = forward(x, w1, w2)
    return (a2[0] - t)**2/2

def central_diff(x, t, w1, w2):
    ## 偏微分係数の計算
    diff1 = zeros((M, D+1))
    diff2 = zeros((K, M+1))

    # 隠れ層
    e1 = zeros((M, D+1)) # 摂動
    for i in range(D+1):
        for j in range(M):
            e1[j, i] = DIFF_EPS
            diff1[j, i] = (E(x, t, w1 + e1, w2) - E(x, t, w1 - e1, w2))\
                    / (2 * DIFF_EPS)
            e1[j, i] = 0

    # 出力層
    e2 = zeros((K, M+1)) # 摂動
    for i in range(M+1):
        for j in range(K):
            e2[j, i] = DIFF_EPS
            diff2[j, i] = (E(x, t, w1, w2 + e2) - E(x, t, w1, w2 - e2))\
                    / (2 * DIFF_EPS)
            e2[j, i] = 0

    return (diff1, diff2)

#=== テストデータ ==]
x = linspace(-1, 1, N)
t = x**2
w1 = random.uniform(-1, 1, (M, D+1))
w2 = random.uniform(-1, 1, (K, M+1))

bdiff1 = zeros((M, D+1))
bdiff2 = zeros((K, M+1))
cdiff1 = zeros((M, D+1))
cdiff2 = zeros((K, M+1))
for i in range(N):
    bd1, bd2 = backpropagation(x[i], t[i], w1, w2)
    cd1, cd2 = central_diff(x[i], t[i], w1, w2)
    bdiff1 += bd1
    bdiff2 += bd2
    cdiff1 += cd1
    cdiff2 += cd2

print u"誤差逆伝播で求めた微分係数"
print append(bdiff1.ravel(), bdiff2.ravel())
print u"中心差分 (eps=%f) で求めた微分係数" % DIFF_EPS
print append(cdiff1.ravel(), cdiff2.ravel())
