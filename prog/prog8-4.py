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

ALPHA = 0.3       # 最急降下法の勾配係数
INIT_COUNT = 100  # 最急降下法によるならし回数
ITER_MAX = 2      # 準ニュートン法の最大反復回数
ITER_EPS = 5.0e-2 # 準ニュートン法の停止パラメータ
HESSIAN_ALPHA = 1.0e-2 # ヘッセ行列の初期値パラメータ

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
    a1 = w1.dot([x, 1])         # 隠れ層への入力
    a2 = w2.dot(append(tanh(a1), 1))  # 出力層への入力
    return (a1, a2)

#=== 誤差逆伝播法 ===
# 各重みに関する偏微分係数を計算
def backpropagation1(x, t, w1, w2):
    a1, a2 = forward(x, w1, w2)  # 順伝播
    delta2 = a2-t                # 出力の誤差
    tanh_a1 = tanh(a1)
    delta1 = (1- tanh_a1**2)*w2[:,0:M].T.dot(delta2) # 隠れ層の誤差

    ## 偏微分係数の計算
    diff1 = zeros((M, D+1))
    diff2 = zeros((K, M+1))

    # 隠れ層
    diff1 = outer(delta1, [x, 1])
    # 出力層
    diff2 = outer(delta2, append(tanh_a1, 1))
    return (diff1, diff2)

def backpropagation2(x, t, w1, w2):
    a1, a2 = forward(x, w1, w2)  # 順伝播
    delta2 = ones(K)             # 出力の誤差
    tanh_a1 = tanh(a1)
    delta1 = (1- tanh_a1**2)*w2[:,0:M].T.dot(delta2) # 隠れ層の誤差

    ## 偏微分係数の計算
    diff1 = zeros((M, D+1))
    diff2 = zeros((K, M+1))

    # 隠れ層
    diff1 = outer(delta1, [x, 1])
    # 出力層
    diff2 = outer(delta2, append(tanh_a1, 1))
    return (diff1, diff2)

#=== 準ニュートン法
def fit(outname, expr, f):
    print expr
    x = linspace(-1, 1, N)
    t = vectorize(f)(x)

    # 再急降下法で適当回数ならし運転
    w1 = random.uniform(-1, 1, (M, D+1))
    w2 = random.uniform(-1, 1, (K, M+1))
    for i in range(INIT_COUNT):
        for j in range(N):
            d1, d2 = backpropagation1(x[j], t[j], w1, w2)
            print d1, d2
            w1 -= ALPHA*d1
            w2 -= ALPHA*d2

    # 準ニュートン法
    w = append(w1.flatten(), w2.flatten())
    for i in range(ITER_MAX):
        finish = True
        # 以降 H は H^(-1) の事
        H = identity(W)/HESSIAN_ALPHA
        diff = zeros(W)
        for j in range(N):
            d1, d2 = backpropagation2(x[j], t[j], w1, w2)
            print d1
            print d2

            if LA.norm(d1) >= ITER_EPS or LA.norm(d2) >= ITER_EPS:
                finish = False

            b = append(d1.flatten(), d2.flatten())
            H -= H.dot(outer(b, b)).dot(H)/(1+b.dot(H.dot(b)))
            diff += b

        if finish: break

        w -= H.dot(diff)
        print w
    count = i

    w1 = w[0:W1].reshape((M, D+1))
    w2 =w[W1:W].reshape((K, M+1))
    
    test_x = linspace(-1, 1, N)
    test_y = vectorize(lambda x: forward(x, w1, w2)[1][0])(test_x)

    xlim(-1, 1)
    scatter(x, t)
    plot(test_x, test_y)
    title("%s (iteration=%d)" % (expr, count))
    savefig("fig8-4-%s.png" % outname)
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
