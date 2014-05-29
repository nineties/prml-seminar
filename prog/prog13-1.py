# -*- coding: utf8 -*-
import random
import numpy as np
from matplotlib.pyplot import *
import matplotlib.cm as cmap

#== 2クラスSVM ==
class SVM:
    # デフォルトセッティング
    # SMO法の反復回数上限
    MAX_ITER = 1000
    # |x|<ZERO_EPS のとき, x=0が成立していると見なす.
    ZERO_EPS = 1.0e-2
    # ソフトマージン最適化のパラメータC(大きい程ハードマージンに近づく)
    SOFTMARGIN_C = 1.0e1
    # カーネル関数
    def gaussian(x1, x2):
        return np.exp(-sum((x1-x2)**2)/(2*0.5**2))

    def __init__(self, max_iter=MAX_ITER, zero_eps=ZERO_EPS, softmargin=SOFTMARGIN_C, kernel=gaussian):
        self.max_iter   = max_iter
        self.zero_eps   = zero_eps
        self.softmargin = softmargin
        self.kernel     = kernel

    def gram_matrix(svm):
        N = svm.N
        K = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                K[i,j] = svm.kernel(svm.train_x[i,:], svm.train_x[j,:])
        return K

    # 定数項を計算
    def threshold(svm):
        N = svm.N
        v = 0
        n = 0
        for i in range(N):
            if svm.mu[i] < svm.zero_eps and\
                    svm.softmargin-svm.mu[i] < svm.zero_eps: continue
            n += 1
            s = svm.train_t[i]
            for j in range(N):
                if svm.mu[j] < svm.zero_eps and\
                        svm.softmargin-svm.mu[j] < svm.zero_eps: continue
                s -= svm.mu[j]*svm.train_t[j]*svm.K[i, j]
            v += s
        return v/n

    # 識別関数
    def discriminant(self, x):
        v = 0
        for i in range(self.N):
            if self.mu[i] < self.zero_eps and\
                    self.softmargin-self.mu[i] < self.zero_eps: continue
            v += self.mu[i]*self.train_t[i]*\
                    self.kernel(self.train_x[i,:], x)
        return v + self.theta

    # KKT条件が成立しているならTrue
    def checkKKT(svm, i):
        yi = svm.train_t[i] * svm.discriminant(train_x[i,:])
        if svm.softmargin - svm.mu[i] < svm.zero_eps:
            return yi >= 1 and svm.mu[i]*(yi-1) < svm.zero_eps
        else:
            return yi <= 1

    # 2つ目の更新ベクトルを選択
    def choose_second(svm, i):
        di = svm.discriminant(svm.train_x[i,:])
        m  = 0
        mj = 0
        for j in range(N):
            if svm.mu[j] < svm.zero_eps: continue
            v = abs(svm.discriminant(train_x[j,:])-di)
            if v > m:
                m = v
                mj = j
        return mj

    # muとthetaを更新
    def update(svm, i, j):
        if i == j: return False
        ti     = svm.train_t[i]
        tj     = svm.train_t[j]
        di     = svm.discriminant(train_x[i,:])
        dj     = svm.discriminant(train_x[j,:])
        deltai = (1-ti*tj+ti*(dj-di))/(svm.K[i,i]-2*svm.K[i,j]+svm.K[j,j])
        c      = ti*svm.mu[i] + tj*svm.mu[j]
        next_mui = svm.mu[i] + deltai
        if ti == tj:
            l = max(0, c/ti-svm.softmargin)
            h = max(svm.softmargin, c/ti)
            if next_mui < l: next_mui = l
            elif next_mui > h: next_mui = h
        else:
            l = max(0, c/ti)
            h = min(svm.softmargin, svm.softmargin+c/ti)
            if next_mui < l: next_mui = l
            elif next_mui > h: next_mui = h
        if abs(next_mui - svm.mu[i]) < svm.zero_eps: return False
        svm.mu[i] = next_mui
        svm.mu[j] = (c-ti*svm.mu[i])/tj
        svm.theta = SVM.threshold(svm)
        return True

    # サポートベクタ以外, グラム行列を除去
    def strip(self):
        idxs = self.mu < self.zero_eps
        self.train_x = self.train_x[idxs]
        self.train_t = self.train_t[idxs]
        self.mu      = self.mu[idxs]
        self.N       = len(self.mu)
        self.K       = None

    # train_x: (N, D)型のfloat型numpy.array
    # train_t: 長さNのint型numpy.array (値は1,-1)
    def learn(self, train_x, train_t):
        #== SMO 法 ==
        # 以下のメインループはアイデアを理解してもらう為に
        # 非常に単純化したものです. 従って遅いです.
        # もっと高速な方法はSMO法の論文を参照してください.
        self.N       = len(train_t)
        self.train_x = np.copy(train_x)
        self.train_t = np.copy(train_t)
        self.mu      = np.random.uniform(0, 1, self.N)
        self.K       = SVM.gram_matrix(self)
        self.theta   = SVM.threshold(self)
        for p in range(self.max_iter):
            changed = False
            for i in range(N):
                if SVM.checkKKT(self, i): continue
                j       = SVM.choose_second(self, i)
                changed = SVM.update(self, i, j) or changed
            if not changed: break
        #self.strip()

    def classify(self, x):
        if self.discriminant(x) > 0:
            return 0
        else:
            return 1

class MultiSVM:
    def __init__(self):
        self.svms = None

    # train_x: (N, D)型のfloat型numpy.array
    # train_t: 長さNのint型numpy.array (値は0..K-1)
    # K      : クラスの数
    def learn(self, train_x, train_t, K):
        N = len(train_t)
        self.svms = []
        for k in range(K):
            print "%d vs. others"%k
            tmp_train_t = np.ones(N, dtype=int)
            tmp_train_t[np.where(train_t != k)] = -1
            svm = SVM()
            svm.learn(train_x, tmp_train_t)
            self.svms.append(svm)

    def classify(self, x):
        svms = self.svms
        return np.argmax([svms[i].discriminant(x) for i in range(len(svms))])

#== 実験 ==

np.random.seed(0)
random.seed(0)

N = 100
K = 4
train_x = np.zeros((N, 2))
train_t = np.zeros(N, dtype=int)
for i in range(N):
    k = random.randint(0, K-1)
    mu = [[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]][k]
    train_x[i, :] = np.random.normal(mu, [0.3, 0.3], 2)
    train_t[i] = k
idxs = np.arange(0, N)
random.shuffle(idxs)
train_x = train_x[idxs]
train_t = train_t[idxs]

msvm = MultiSVM()
msvm.learn(train_x, train_t, K)

xlim(-1,1)
ylim(-1,1)
X, Y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
Z = np.vectorize(lambda x, y: msvm.classify([x,y]))(X, Y)
scatter(train_x[:,0], train_x[:,1], c=train_t, s=50, cmap=cmap.cool)
pcolor(X, Y, Z, alpha=0.2)
show()
clf()

#for i in range(K):
#    xlim(-1,1)
#    ylim(-1,1)
#    X, Y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
#    svm = msvm.svms[i]
#    Z = np.vectorize(lambda x, y: svm.discriminant([x,y]))(X, Y)
#    scatter(train_x[:,0], train_x[:,1], c=train_t, s=50, cmap=cmap.gist_rainbow)
#    pcolor(X, Y, Z, alpha=0.3)
#    show()
#    clf()
