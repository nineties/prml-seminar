# -*- coding: utf-8 -*-
from numpy import *
from scipy import linalg as LA
from matplotlib.pyplot import *

N = 50

D = 1   # 入力の次元
M = 3   # 隠れ層の数
K = 1   # 出力層の数

ALPHA = 0.3       # 最急降下法の勾配係数
ITER_MAX = 5000   # 最急降下法の反復回数上限
ITER_EPS = 5.0e-2 # 再急降下法の停止パラメータ

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

    # 逆伝播
    delta1 = ((1-tanh_a1**2)*w2[:,0:M]).T.dot(delta2) # 隠れ層の誤差

    ## 偏微分係数の計算
    diff1 = zeros((M, D+1))
    diff2 = zeros((K, M+1))
    diff1 = outer(delta1, append(x, 1))
    diff2 = outer(delta2, append(tanh_a1, 1))
    return (diff1, diff2)

#=== 学習データ ===
x = linspace(-1, 1, N)
t = x**2

w1 = random.uniform(-1, 1, (M, D+1))
w2 = random.uniform(-1, 1, (K, M+1))
for i in range(ITER_MAX):
    finish = True
    for j in range(N):
        d1, d2 = backpropagation(x[j], t[j], w1, w2)
        w1 -= ALPHA*d1
        w2 -= ALPHA*d2
        if LA.norm(d1) >= ITER_EPS or LA.norm(d2) >= ITER_EPS:
            finish = False
    if finish: break
count = i

#=== ヤコビ行列 ===
# 入力に対する出力の偏微分係数を計算
def jacobian(x, w1, w2):
    a1, a2 = forward(x, w1, w2)  # 順伝播
    delta2 = ones(K)             # 出力の誤差
    tanh_a1 = tanh(a1)
    delta1 = ((1- tanh_a1**2)*w2[:,0:M]).T.dot(delta2) # 隠れ層の誤差

    # ヤコビ行列の計算
    return w1[:,0].dot(delta1)

test_x = linspace(-1, 1, N)
test_y = vectorize(lambda x: forward(x, w1, w2)[1][0])(test_x)
test_diffy = vectorize(lambda x: jacobian(x, w1, w2))(test_x)

xlim(-1, 1)
ylim(-2, 2)
scatter(x, t)
plot(test_x, test_y, label="y=x^2")
plot(test_x, test_diffy, label="jacobian matrix")
plot(test_x, 2*test_x, label="y=2x")
title("y=x^2 (iteration=%d)" % count)
legend(loc=4)
savefig("fig8-2.png")
