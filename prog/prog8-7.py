# -*- coding: utf-8 -*-
from numpy import *
from scipy import linalg as LA
from matplotlib.pyplot import *
import matplotlib.cm as cm

N = 200

D = 2   # 入力の次元
M = 4   # 隠れ層の数
K = 1   # 出力層の数

W1  = M*(D+1) # 第1層の重みパラメータ数
W2  = K*(M+1) # 第2層の重みパラメータ数
W   = W1 + W2 # 重みパラメータ数

QN_ITER_MAX = 100    # 準ニュートン法の最大反復回数
QN_ITER_EPS = 1.0e-2 # 準ニュートン法の停止パラメータ
QN_HESSE0 = 1.0e-2   # ヘッセ行列の初期値パラメータ

# 直線探索のパラメータ
LS_C1 = 1.0e-4
LS_C2 = 0.9
LS_SCALE_UP   = 2.0
LS_SCALE_DOWN = 0.6
LS_ITER_MAX = 100
LS_STEP0 = 2

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

#=== 準ニュートン法の1ステップ

def sigmoid(a):
    return 1/(1+exp(-a))

# 誤差関数と勾配
def error_grad(x, t, w1, w2):
    E = 0
    gradE1 = 0
    gradE2 = 0
    for i in range(N):
        a1, a2 = forward(x[i], w1, w2)
        y = sigmoid(a2[0])
        d1, d2 = diffcoef(x[i], a1, a2, w1, w2, y-t[i])
        E -= t[i]*log(y) + (1-t[i])*log(1-y) # 交差エントロピー誤差関数
        gradE1 += d1
        gradE2 += d2
    return (E, gradE1, gradE2)

# 直線探索
# w1, w2: 現在の位置 (重み)
# p1, p2: 探索方向
def line_search(x, t, w1, w2, p1, p2):
    # 方向ベクトルを正規化した状態での初期stepを設定
    # 傾き緩やかなほど大きく移動.
    step = LS_STEP0/(LA.norm(p1) + LA.norm(p2))

    E, gradE1, gradE2 = error_grad(x, t, w1, w2)
    for c in range(LS_ITER_MAX):
        new_w1 = w1 + step*p1
        new_w2 = w2 + step*p2
        new_E, new_gradE1, new_gradE2 = error_grad(x, t, new_w1, new_w2)

        # armijoの条件
        if new_E > E + LS_C1*step*(sum(p1*gradE1) + sum(p2*gradE2)):
            # stepは大きすぎる
            step *= LS_SCALE_DOWN
            continue

        # wolfeの条件
        if sum(p1*new_gradE1)+sum(p2*new_gradE2) <\
                LS_C2*(sum(p1*gradE1) + sum(p2*gradE2)):
            # stepは小さすぎる
            step *= LS_SCALE_UP
        break
    return step

def quasi_newton_step(x, t, w1, w2):
    # 誤差関数のa1,a2での微分係数
    diff1 = zeros((M, D+1))
    diff2 = zeros((K, M+1))

    # B = H^-1 (ヘッセ行列の逆行列)
    B = identity(W)/QN_HESSE0

    for i in range(N):
        a1, a2 = forward(x[i], w1, w2)
        y = sigmoid(a2[0])

        d1, d2 = diffcoef(x[i], a1, a2, w1, w2, y-t[i])
        diff1 += d1
        diff2 += d2

        d1, d2 = diffcoef(x[i], a1, a2, w1, w2, y*(1-y))
        b = append(d1.ravel(), d2.ravel())
        B -= outer(B.dot(b), b).dot(B)/(1 + b.dot(B.dot(b)))

    delta_w = - B.dot(append(diff1.ravel(), diff2.ravel()))
    delta_w1 = delta_w[0:W1].reshape((M, D+1))
    delta_w2 = delta_w[W1:W].reshape((K, M+1))
    return (delta_w1, delta_w2)

def quasi_newton_method(x, t):
    w1 = random.uniform(-1, 1, (M, D+1))
    w2 = random.uniform(-1, 1, (K, M+1))
    for i in range(QN_ITER_MAX):
        d1, d2 = quasi_newton_step(x, t, w1, w2)
        step = line_search(x, t, w1, w2, d1, d2)
        w1 += step*d1
        w2 += step*d2
        if LA.norm(d1) < QN_ITER_EPS and LA.norm(d2) < QN_ITER_EPS:
            break
    return (w1, w2, i+1)

#=== 2クラス識別実験 ===

x = random.uniform(-1, 1, (N, 2))
t = logical_xor(x[:,0]**2 + x[:,1]**2 < 0.6, x[:,0] < x[:,1])

w1, w2, count = quasi_newton_method(x, t)
print count

X, Y = meshgrid(linspace(-1, 1, 100), linspace(-1, 1, 100))
Z = vectorize(lambda x,y: sigmoid(forward([x,y], w1, w2)[1][0]))(X, Y)
xlim(-1, 1)
ylim(-1, 1)
scatter(x[:,0], x[:,1], c=t, s=50, cmap=cm.cool)
pcolor(X, Y, Z, alpha=0.3)
title("2-class logistic regression")
savefig("fig8-7.png")
