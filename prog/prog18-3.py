# -*- coding: utf-8 -*0
from numpy import *
from matplotlib.pyplot import *
import matplotlib.cm as cm

# K-meansアルゴリズムの実装例

N = 1500
x = random.uniform(-5, 5, (N, 2))
#x = concatenate([
#    random.multivariate_normal([0, 0], eye(2), N/3),
#    random.multivariate_normal([0, 5], eye(2), N/3),
#    random.multivariate_normal([2, 3], eye(2), N/3),
#    ])

K = 30

# 代表ベクトル(ランダムにK個選択)
mu = random.multivariate_normal([0, 0], eye(2), K)

# 各ベクトルの所属クラス(0,1,...,K-1)
cls = zeros(N, dtype=int)

for it in range(100):
    # [E-step] 所属クラスの更新

    # 距離を計算して
    distance = [sum((x - mu[k,:])**2, axis=1) for k in range(K)]

    # 最小の所に割り当て
    cls = argmin(distance, axis=0)

    # [M-step] 代表ベクトルの更新
    new_mu = array([x[cls == k].mean(0) for k in range(K)])

    # 代表ベクトルが動かなくなったら終了
    if max((sum((new_mu - mu)**2, axis=1))) < 1.0e-2:
            break

    # 途中結果の図示
    clf()
    title('K-means clustering (iteration=%d)' % (it+1))
    xlim(-5, 5)
    ylim(-5, 5)
    scatter(x[:, 0], x[:, 1], c = cls, s=50, cmap=cm.cool)
    for k in range(K):
        text(mu[k, 0]+0.1, mu[k, 1]+0.1, 'center of cluster %d' % k)
        plot(mu[k, 0], mu[k, 1], "bo", color='black')
        plot(new_mu[k, 0], new_mu[k, 1], "bo", color='red')
        #arrow(mu[k, 0], mu[k, 1], new_mu[k, 0]-mu[k, 0], new_mu[k, 1]-mu[k, 1], color="red", width=0.01, length_includes_head=True)
    savefig('fig18-3-2-%d.png' % it)

    mu = new_mu

