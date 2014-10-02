# -*- coding: utf-8 -*-
from numpy import *
from matplotlib.pyplot import *
import scipy.stats as stats
import re

#=== データの準備 ===
# UJI/ に
# http://archive.ics.uci.edu/ml/datasets/UJI+Pen+Characters
# からダウンロードしたファイルを置く

def read_traj_data(person, char):
    # 一人あたり2データ
    traj = [[], []]
    f = open('UJI/UJIpenchars-w%02d' % person)
    # .SEGMENT CHARACTER ... という行を見つける
    pat = re.compile('.SEGMENT CHARACTER \d+ \? "%s"' % char)
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

# 11人分布の'a'の手書きデータを読み出す
data = []
for person in range(1, 11+1):
    data += read_traj_data(person, 'a')

#=== 学習 ===
#== 隠れ状態数 ==
K = 16

#== 遷移行列 ==
# left-to-right HMM なので transition matrix は
# A[k, k] と A[k, k+1] 以外は0
# 初期化は A[k, k] = 1/2, A[k, k+1] = 1/2 とする.
# 但し，最終状態から次への遷移はないので A[K-1, K-1] = 1
A = zeros((K, K))
for k in range(K-1):
    A[k, k] = 0.5
    A[k, k+1] = 0.5
A[K-1, K-1] = 1

#== emission分布のパラメータ ==
xmin, xmax, ymin, ymax = (lambda d: (min(d[:, 0]), max(d[:, 0]), min(d[:, 1]), max(d[:, 1])))(concatenate(data))

mu = zeros((K, 2))
mu[:, 0] = random.uniform(xmin, xmax, K)
mu[:, 1] = random.uniform(ymin, ymax, K)
sigma = zeros((K, 2, 2))
sigma[:] = 1.0e5*eye(2)

# EM法のメインループ
for it in range(10000):
    print it
    # E step

    # 各系列毎に以下のパラメータを計算
    gamma = []
    sum_xi = zeros((K, K)) # xiは和を取ってしまう

    # 各データ毎にforward-backward algorithm
    for x in data:
        #== forward-backward algorithm ==
        # under flowを避ける為，p(x) を低数倍した
        # 2*pi*|sigma|*p(x) = exp(- (x-mu)^T sigma^-1 (x-mu)/2)
        # を使う.
        xmu = x.reshape(-1, 1, 2) - mu.reshape(1, -1, 2)
        emission = exp(-einsum('ikp,kpq,ikq->ik', xmu, linalg.inv(sigma), xmu)/2)

        N = len(x)
        alpha = zeros((N, K))
        alpha[0, 0] = emission[0, 0]
        for i in xrange(1, N):
            alpha[i] = emission[i]*alpha[i-1].dot(A)

        beta  = zeros((N, K))
        beta[N-1] = 1
        for i in reversed(xrange(N-1)):
            beta[i] = A.dot(beta[i+1]*emission[i+1])
        
        # gamma(i), sum_i xi(i)の計算
        px = alpha[N-1].sum()
        gamma.append(alpha*beta/px)
        sum_xi += alpha[:-1].T.dot(beta[1:]*emission[1:])*A/px

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

    # sigmaの更新
    new_sigma = zeros((K, 2, 2))
    for i, x in enumerate(data):
        xmu = x.reshape(-1, 1, 2) - new_mu.reshape(1, -1, 2)
        new_sigma += einsum('ik,ikp,ikq->kpq', gamma[i], xmu, xmu)
    new_sigma /= sum_gamma.reshape(-1, 1, 1)

    if linalg.norm(new_mu - mu) / linalg.norm(mu) < 1.0e-30:
        break

    A = new_A
    mu = new_mu
    sigma = new_sigma

clf()
gca().invert_yaxis()
scatter(mu[:, 0], mu[:, 1])
plot(mu[:, 0], mu[:, 1])
show()
