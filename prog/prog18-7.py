# -*- coding: utf-8 -*0
from numpy import *
from matplotlib.pyplot import *
import matplotlib.cm as cm
import scipy.stats as stats

# 混合ガウス分布クラスタリングの実装例

N = 150
x = concatenate([
    random.multivariate_normal([0, 0], eye(2), N/3),
    random.multivariate_normal([0, 5], eye(2), N/3),
    random.multivariate_normal([2, 3], eye(2), N/3),
    ])

K = 3

# 各パラメータを初期化
mu = [random.multivariate_normal([0, 0], eye(2)) for k in range(K)]
S  = [0.1*eye(2) for k in range(K)] # 対角行列で初期化
pi = ones(K)/K # 一様分布で初期化

# 各ベクトルの各クラスへの所属確率
r = zeros((N, K))

for it in range(100):
    # [E-step]
    for c in range(K):
        r[:, c] = pi[c] * stats.multivariate_normal(mu[c], S[c]).pdf(x)
    r = r / repeat(r.sum(1), K).reshape(-1,K)

    # [M-step]
    changed = False
    for c in range(K):
        new_mu = r[:, c].dot(x)/r[:, c].sum()
        if linalg.norm(new_mu - mu[c]) > 1.0e-3:
            changed = True
        mu[c] = new_mu
    for c in range(K):
        rr = repeat(r[:,c], 2).reshape(-1, 2)
        S[c] = (rr * (x-mu[c])).T.dot((x-mu[c])) / r[:, c].sum()
    for c in range(K):
        pi[c] = r[:, c].sum() / N

    # 平均が動かなくなったら終了(本当は他も見たほうが良いけど)
    if not changed:
        break

    # 途中結果の図示
    clf()
    xlim(-3, 5)
    ylim(-3, 8)
    cls = argmax(r, axis=1)
    scatter(x[:, 0], x[:, 1], c = cls, s=50)
    X, Y = meshgrid(linspace(-3, 5), linspace(-3, 8))
    Z = vectorize(lambda x,y: sum([pi[c]*stats.multivariate_normal(mu[c],S[c]).pdf([x,y]) for c in range(K)]))(X, Y)
    pcolor(X, Y, Z, alpha=0.2)
    savefig('fig18-7-%d.png' % it)
print 'iteration =', it
