# -*- coding: utf-8 -*-
from numpy import *
from matplotlib.pyplot import *
from scipy.misc import logsumexp
from matplotlib.patches import Ellipse
import re
import time

#=== データの準備 ===
# UJI/ に
# http://archive.ics.uci.edu/ml/datasets/UJI+Pen+Characters
# からダウンロードしたファイルを置く

def read_traj_data(person, char):
    # 一人あたり2データ
    traj = [[], []]
    f = open('UJI/UJIpenchars-w%02d' % person)
    # .SEGMENT CHARACTER ... という行を見つける
    pat = re.compile('.SEGMENT CHARACTER[^?]*\? "%s"' % char)
    cnt = 0
    while True:
        line = f.readline()
        if not line: break
        result = pat.search(line)
        if result:
            f.readline()
            f.readline()
            f.readline()
            while True:
                line = f.readline().strip()
                if line == '.PEN_UP':
                    break
                traj[cnt].append( map(float, line.split()) )
            traj[cnt] = array(traj[cnt])
            cnt += 1
    f.close()
    return traj

# 11人分の'2'の手書きデータを読み出す
data = []
for person in range(1, 11+1):
    data += read_traj_data(person, '2')

#=== 学習 ===
#== 隠れ状態数 ==
K = 16

#== 遷移行列 ==
# 単純な left-to-right HMM
# 初期状態は常に0
# transition matrix は
# A[k, k] と A[k, k+1] 以外は0
# 初期化は A[k, k] = 1/2, A[k, k+1] = 1/2 とする.
# 但し，最終状態から次への遷移はないので A[K-1, K-1] = 1

init = zeros(K)
init[0] = 1

A = zeros((K, K))
for k in range(K-1):
    A[k, k] = 0.5
    A[k, k+1] = 0.5
A[K-1, K-1] = 1

#== emission分布のパラメータ ==
mu = random.uniform(0, 1000, (K, 2))
sigma = 300

for it in range(10000):
    # E step
    print it

    # 各系列毎に以下のパラメータを計算
    gamma = []
    sum_xi = zeros((K, K)) # xiは和を取ってしまう

    # 各データ毎にforward-backward algorithm
    for x in data:
        N = len(x)

        #== forward-backward algorithm ==
        # emissionをあらかじめ計算
        em = exp(-0.5*sum(((x.reshape(-1, 1, 2) - mu.reshape(1, -1, 2)))**2, axis=2)/sigma**2)
        em /= 2*pi*sigma**2

        # Scaling factor
        C = zeros(N)

        alpha = zeros((N, K))
        alpha[0] = em[0] * init
        C[0] = alpha[0].sum()
        alpha[0] /= C[0]
        for i in xrange(1, N):
            alpha[i] = em[i]*alpha[i-1].dot(A)
            C[i] = alpha[i].sum()
            alpha[i] /= C[i]

        beta = zeros((N, K))
        beta[N-1] = 1
        for i in reversed(xrange(N-1)):
            beta[i] = A.dot(beta[i+1]*em[i+1])/C[i+1]

        # gamma(i), sum_i xi(i)の計算
        gamma.append( alpha*beta )
        sum_xi += alpha[:-1].T.dot(c_[C[1:]]*beta[1:]*em[1:])*A

    # M step
    # A の更新
    new_A = sum_xi / c_[sum_xi.sum(1)]

    # muの更新
    new_mu = zeros((K, 2))
    sum_gamma = zeros(K)
    for i, x in enumerate(data):
        sum_gamma += gamma[i].sum(0)
        new_mu += gamma[i].T.dot(x)
    new_mu /= c_[sum_gamma]

    if linalg.norm(new_mu - mu) < 1.0e-10 * linalg.norm(mu):
        break

    A = new_A
    mu = new_mu

clf()
gca().invert_yaxis()
scatter(mu[:, 0], mu[:, 1])
plot(mu[:, 0], mu[:, 1])
show()
