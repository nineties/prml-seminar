# -*- coding: utf-8 -*-
from numpy import *
from scipy import linalg as LA
from matplotlib.pyplot import *

N = 50

D = 1   # 入力の次元
M = 3   # 隠れ層の数
K = 1   # 出力層の数

W1  = M*(D+1) # 第1層の重みパラメータ数
W2  = K*(M+1) # 第2層の重みパラメータ数
W   = W1 + W2 # 重みパラメータ数

ALPHA = 0.01      # 最急降下法の勾配係数
ITER_MAX = 5000   # 最大反復回数
ITER_EPS = 1.0e-3 # 勾配のノルムがこれかになったら停止

#=== 重みパラメータ ===
# 隠れ層の重みは M*(D+1) 行列w1で表現.
# 出力層の重みは K*(M+1) 行列w2で表現.
#
# w1[i, j] はj番目の入力と隠れ層のi番目の素子の間の重み.
# w2[i, j] は隠れ層のj番目素子と出力層のi番目の素子の間の重み.

#=== 順伝播 ===
# x: 入力
# w1: 隠れ層の重み
# w2: 出力層の重み
#
# 戻り値はタプル (a1, a2)
# a1[i]: 隠れ層iへの入力
# a2[i]: 出力層iへの入力
def forward(x, w1, w2):
    a1 = w1.dot(append(x, 1))         # 隠れ層への入力
    a2 = w2.dot(append(tanh(a1), 1))  # 出力層への入力
    return (a1, a2)

#=== 誤差逆伝播 ===
# a1, a2: 各層への入力
# w1, w2: 各層の重み
# delta2: 出力層の誤差
# 戻り値: 隠れ層の誤差
def backprop(a1, a2, w1, w2, delta2):
    return ((1- tanh(a1)**2)*w2[:,0:M]).dot(delta2) # 隠れ層の誤差

# 偏微分係数の計算
def diffcoef(x, a1, a2, w1, w2, delta2):
    delta1 = backprop(a1, a2, w1, w2, delta2)
    diff1 = outer(delta1, append(x, 1))
    diff2 = outer(delta2, append(tanh(a1), 1))
    return (diff1, diff2)

#=== 最急降下法の1ステップの更新 ===
def steepest_step(x, t, w1, w2):
    diff1 = zeros((M, D+1))
    diff2 = zeros((K, M+1))
    for i in range(N):
        a1, a2 = forward(x[i], w1, w2)
        # i番目のデータに対する誤差関数の a1, a2 での微分係数
        d1, d2 = diffcoef(x[i], a1, a2, w1, w2, a2-t[i])
        diff1 -= ALPHA*d1
        diff2 -= ALPHA*d2
    return (diff1, diff2)

# 最急降下法
# 学習結果 w1, w2 と反復回数を返す
def steepest_descent_method(x, t):
    w1 = random.uniform(-1, 1, (M, D+1))
    w2 = random.uniform(-1, 1, (K, M+1))
    for i in range(ITER_MAX):
        d1, d2 = steepest_step(x, t, w1, w2)
        w1 += d1
        w2 += d2
        if LA.norm(d1) < ITER_EPS and LA.norm(d2) < ITER_EPS:
            break
    return (w1, w2, i+1)

#=== 実験 ===
def fit(outname, expr, f):
    print expr
    x = linspace(-1, 1, N)
    t = vectorize(f)(x)
    xlim(-1, 1)
    scatter(x, t)

    # 再急降下法
    w1, w2, count = steepest_descent_method(x, t)
    y = vectorize(lambda x: forward(x, w1, w2)[1][0])(x)
    plot(x, y)

    title("%s (iteration=%d" % (expr, count))
    savefig("fig7-1-%s.png" % outname)
    clf()

fit("quadratic", "y=x^2", lambda x: x**2)
fit("sin", u"y=sin(πx)", lambda x: sin(pi*x))
fit("abs", "y=|x|", abs)

def heaviside(x):
    if x < 0:
        return 0
    else:
        return 1

fit("heaviside", "y=H(x) (Heaviside function)", heaviside)
