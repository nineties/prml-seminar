# -*- coding: utf-8 -*-
from numpy import *
from matplotlib.pyplot import *

N = 50

D = 1   # 入力の次元
M = 3   # 隠れ層の数
K = 1   # 出力層の数

#=== 重みパラメータ ===
w1 = zeros((M, D+1))    # 隠れ層の入力の重み
w2 = zeros((K, M+1))    # 出力層の入力の重み

#=== 順伝播 ===
# 入力 x に対して各素子への入力を計算する.
# 出力はタプル (a1, a2) であり,
# a1[i]が隠れ層の素子 i への入力, a2[i]が出力層の素子 i への入力

def forward(x):
    a1 = w1.dot([x, 1])         # 隠れ層への入力
    a2 = w2.dot(append(a1, 1))  # 出力層への入力
    return (a1, a2)

#=== 誤差逆伝播法 ===
# 各重みに関する偏微分係数を計算
def backpropagation(x, t):
    a1, a2 = forward(x)  # 順伝播
    delta2 = a2 - t      # 出力の誤差
    delta1 = (1- tanh(a1)**2)*w2[:,0:M].T.dot(delta2) # 隠れ層の誤差

    # 偏微分係数の計算
    d1 = zeros((M, D+1))
    d2 = zeros((K, M+1))
    # 隠れ層
    for i in range(M):
        for j in range(D+1):
            if j <= D:
                d1[i,j] = tanh(a1[j])*delta1[i]
            else:
                d1[i,j] = delta1[i]
    # 出力層
    for i in range(K):
        for j in range(M+1):
            if j <= M:
                d2[i,j] = tanh(a2[j])*delta2[i]
            else:
                d2[i,j] = delta2[i]

    return (d1, d2)

print backpropagation(1, 1)

#=== 学習データ ===
#x = random.uniform()
#t = x**2
#
#scatter(x, t)
#title("Training data")
#savefig("fig7-1-training.png")
#


