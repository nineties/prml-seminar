# -*- coding: utf8 -*-
import random
import numpy as np
from matplotlib.pyplot import *
import matplotlib.cm as cmap

#== 2クラスSVMRegress ==
class SVMRegress:
    # デフォルトセッティング
    # SMO法の反復回数上限
    MAX_ITER = 500
    # |x|<ZERO_EPS のとき, x=0が成立していると見なす.
    ZERO_EPS = 1.0e-1
    # SVMRegress回帰のパラメータC(大きくするとフィッティングの精度が上がる)
    REGRESS_C   = 1.0e5
    # 誤差関数のパラメータ(小さくするとより厳密. 大きくするとよりスパース)
    REGRESS_EPS = 0.8
    # カーネル関数
    def gaussian(x1, x2):
        return np.exp(-(x1-x2)**2/(2*0.3**2))

    def __init__(self, max_iter=MAX_ITER, zero_eps=ZERO_EPS,\
            eps=REGRESS_EPS, C=REGRESS_C, kernel=gaussian):
        self.max_iter = max_iter
        self.zero_eps = zero_eps
        self.C        = C
        self.eps      = eps
        self.kernel   = kernel

    def gram_matrix(svm):
        N = svm.N
        K = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                K[i,j] = svm.kernel(svm.train_x[i], svm.train_x[j])
        return K

    # 定数項を計算
    def threshold(svm):
        N = svm.N
        v = 0
        n = 0
        for i in range(N):
            if abs(svm.v[i]) < svm.zero_eps: continue
            n += 1
            if svm.v[i] >= 0:
                s = svm.train_t[i] - svm.eps
            else:
                s = svm.train_t[i] + svm.eps
            for j in range(N):
                if abs(svm.v[j]) < svm.zero_eps: continue
                s -= svm.v[j]*svm.K[i,j]
            v += s
        return v/n

    # 予測値
    def predict(self, x):
        v = 0
        for i in range(self.N):
            if abs(self.v[i]) < self.zero_eps: continue
            v += self.v[i] * self.kernel(self.train_x[i], x)
        return v + self.theta

    # KKT条件が成立しているならTrue
    def checkKKT(svm, i):
        vi   = svm.v[i]
        a    = (abs(vi)+vi)/2
        ahat = (abs(vi)-vi)/2
        # a, ahatのどちらかは0でなければならない
        if a >= svm.zero_eps and ahat >= svm.zero_eps:
            return False
        if a < svm.zero_eps and ahat < svm.zero_eps:
            return True
        if a >= svm.zero_eps:
            # a > 0, ahat = 0
            if svm.C-a < svm.zero_eps:
                return True
            yi = svm.predict(svm.train_x[i])
            if abs(-svm.train_t[i]+yi+svm.eps) < svm.zero_eps:
                return True
            return False
        else:
            # a = 0, ahat > 0
            if svm.C-ahat < svm.zero_eps:
                return True
            yi = svm.predict(svm.train_x[i])
            if abs(svm.train_t[i]-yi+svm.eps) < svm.zero_eps:
                return True
            return False

    # 2つ目の更新ベクトルを選択
    def choose_second(svm, i):
        fi = svm.predict(train_x[i])
        m  = 0
        mj = 0
        for j in range(N):
            if abs(svm.v[i]) < svm.zero_eps: continue
            v = abs(svm.predict(train_x[j])-fi)
            if v > m:
                m = v
                mj = j
        return mj

    # muとthetaを更新
    def update(svm, i, j):
        if i == j: return False
        ti  = svm.train_t[i]
        tj  = svm.train_t[j]
        fi  = svm.predict(train_x[i])
        fj  = svm.predict(train_x[j])
        eta = svm.K[i,i]-2*svm.K[i,j]+svm.K[j,j]
        for sgn in [-2, 0, 2]:
            deltai = (fj-fi+ti-tj-svm.eps*sgn)/eta
            if 2*(svm.v[i] + deltai >= 0) - 2*(svm.v[j] - deltai >= 0)\
                    == sgn: break
        c       = svm.v[i] + svm.v[j]
        next_vi = svm.v[i] + deltai

        # クリッピング
        h       = min(svm.C, c+svm.C)
        l       = max(-svm.C, c-svm.C)
        if next_vi > h: next_vi = h
        elif next_vi < l: next_vi = l

        if abs(next_vi - svm.v[i]) < svm.zero_eps: return False
        svm.v[i] = next_vi
        svm.v[j] = c - next_vi
        svm.theta = SVMRegress.threshold(svm)
        return True

    # train_x[i] (N, D)型のfloat型numpy.array
    # train_t: 長さNのint型numpy.array (値は1,-1)
    def learn(self, train_x, train_t):
        #== SMO 法 ==
        # 以下のメインループはアイデアを理解してもらう為に
        # 非常に単純化したものです. 従って遅いです.
        # もっと高速な方法はSMO法の論文を参照してください.
        self.N       = len(train_t)
        self.train_x = np.copy(train_x)
        self.train_t = np.copy(train_t)
        self.v       = np.random.uniform(0, 1, self.N)
        self.K       = SVMRegress.gram_matrix(self)
        self.theta   = SVMRegress.threshold(self)
        for p in range(self.max_iter):
            changed = False
            for i in range(N):
                if SVMRegress.checkKKT(self, i): continue
                j       = SVMRegress.choose_second(self, i)
                changed = SVMRegress.update(self, i, j) or changed
            if not changed: break
#== 実験 ==

# 学習データ
np.random.seed(0)
N = 20
train_x = np.random.uniform(-1, 1, N)
train_t = np.random.normal(np.sin(2*np.pi*train_x), 0.5)

svm = SVMRegress()
svm.learn(train_x, train_t)

x = np.linspace(-1, 1, 100)
y = np.vectorize(svm.predict)(x)

xlim(-1, 1)
ylim(-2, 2)
count = 0
for i in range(N):
    color = "blue"
    if abs(svm.v[i]) >= svm.zero_eps:
        count += 1
        color = "red"
    plot(train_x[i], train_t[i], "o", color=color)
plot(x, y)
show()

print "%d/%d" % (count, N)
