# -*- coding: utf-8 -*-
from numpy import *
from matplotlib.pyplot import *
import os
import scipy.stats as stats
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from zipfile import ZipFile
from scipy.misc import logsumexp

# 状態数は7
K = 7

# UCI Machine Learning Repositoryの
# http://archive.ics.uci.edu/ml/datasets/Activity+Recognition+from+Single+Chest-Mounted+Accelerometer
# をダウンロードし同じディレクトリに置く
NAME = 'Activity Recognition from Single Chest-Mounted Accelerometer'
ar = ZipFile(open(NAME + '.zip'))
DROP = [0, 2500, 600, 2000, 0, 2500, 1000, 3000, 2000, 3000, 1500]  # 教師ラベルのずれ幅

def read_data(i):
    df = pd.read_csv(ar.open(NAME + '/%d.csv' % (i+1)), header=None)[[1,2,3,4]]
    df.columns = ['x', 'y', 'z', 'label']

    X = df[['x', 'y', 'z']].as_matrix().astype(float)
    t = df['label'].as_matrix()
    if DROP[i] > 0:
        # 加速度データと教師ラベルのズレを修正
        X = X[DROP[i]:]
        t = t[:-DROP[i]]

    # 最初の10秒を落とす
    X = X[5*60*52:]
    t = t[5*60*52:]

    # == 前処理 ==
    # 元データはキャリブレーションなしの生データなので処理が必要.
    # データを取った状況が分からないので以下の処理が妥当かは分かりません.
    #
    # ユーザーの動きによる加速度(AC成分)をa(i), 静的加速度(DC成分)をg(i)
    # センサーの出力をx(i), スケールファクターをs, 
    # 空間座標からセンサー座標への変換行列をT(i)(直交行列)とします.
    # それ以外の要因は考えない事にして
    # X(i) = sT(i)(a(i)+g(i))
    # が成立していると考えます.
    # (センサーの精度次第ですが)gはセンサーの高さで変わります.
    # sはセンサー毎の定数です.
    # Tは姿勢によって変わります.

    # 今興味があるのはa(i)ですので，T(i)やg(i)の影響を取り除く必要があります.
    # 多分被験者は全員同じセンサーを使っていると思うのでsの影響は考えない事に
    # します.

    # データ列を1秒毎(52フレーム毎)にわけて処理します.
    # 1単位時間の間はT(i),g(i)は定数だと仮定すると
    # X(i) = sT(a(i)+g)
    # となるので，平均 E[X(i)] = sT(E[a(i)]+g) を引けば
    # X(i) - E[X(i)] = sT(a(i) - E[a(i)])
    # とgを除去する事が出来ます. またTは直交行列だからノルムを取ると
    # |X(i)-E[X(i)]|^2 = s^2|a(i)-E[a(i)]|^2
    # となってTの影響も除去出来ます.

    N = len(X)/52
    tmp = X[:N*52].reshape(-1, 52, 3)
    tmp = linalg.norm((tmp - tmp.mean(1).reshape(-1, 1, 3)).reshape(-1, 3), axis=1)**2

    # 52Hzのデータは多すぎるので1Hz毎に減らします.
    # これを学習データとして使う事にします.
    x = tmp[:N*52].reshape(-1, 52).mean(1)
    t, _ = stats.mode(t[:N*52].reshape(-1, 52), axis=1)
    t = t.ravel().astype(int)
    return x, t

# 各ラベル毎の加速度データ
data = [ zeros(0) for k in range(K) ]

# 頻度カウント用の配列
A  = zeros((K, K)) # 状態jからkに移行した回数
pi = zeros(K)      # 初期状態がkだった回数

# 10人分を学習
figure(figsize=(18, 6))
for i in range(10):
    x, t = read_data(i)

    # ラベル毎に分ける
    for k in range(K):
        data[k] = append(data[k], x[t==k+1])

    # 頻度カウントの更新
    pi[t[0]] += 1
    for j in range(len(t)-1):
        if t[j] == 0 or t[j+1] == 0: continue
        A[t[j]-1, t[j+1]-1] += 1

    plot(x, label=u'data %d' % i)
legend()
savefig('prog22-1-1.png')

# 初期状態と遷移行列
pi /= pi.sum()
A /= c_[A.sum(1)]

# emission分布はガンマ分布とする
shape_params = []
scale_params = []
for k in range(K):
    shape, _, scale = stats.gamma.fit(data[k], floc=0)
    shape_params.append(shape)
    scale_params.append(scale)

# 学習結果の出力
state_names = ['Working at Computer', 'Standing Up, Walking and Going updown stairs', 'Standing', 'Walking', 'Going UpDown Stairs', 'Walking and Talking with Someone', 'Talking while Standing']

print '== initial state =='
for k in range(K):
    print 'p(z1 = %s) = %.3e' % (state_names[k], pi[k])

print '== transition matrix =='
print '=> prog22-1-trans.[dot, png]'
with open('prog22-1-trans.dot', 'w') as f:
    f.write('digraph {\n')
    for k in range(K):
        f.write('\tnode%d [label="%s"];\n' % (k, state_names[k]))
    for j in range(K):
        for k in range(K):
            if A[j, k] == 0: continue
            f.write('\tnode%d -> node%d [label="%.5f"];\n' % (j, k, A[j, k]))
    f.write('}\n')
os.system('dot -Tpng prog22-1-trans.dot > prog22-1-trans.png')

print '== emission probabilities =='
print '=> prog22-1-1.png'
clf()
figure(figsize=(12, 8))
x = linspace(0, 20000, 1000)
ylim(0, 0.0002)
for k in range(K):
    print state_names[k]
    print 'shape=%.3f' % shape_params[k]
    print 'scale=%.3f' % scale_params[k]
    plot(x, stats.gamma(shape_params[k], loc=0, scale=scale_params[k]).pdf(x), label=state_names[k])
legend()
savefig('prog22-1-2.png')

# 11人目のセンサーデータ
clf()
figure(figsize=(18, 6))
x, t = read_data(10)
plot(x)
savefig('prog22-1-3.png')

# Viterbiアルゴリズム
N = len(t)
omega = zeros((N, K))
pred  = zeros((N, K))
for k in range(K):
    omega[0, k] = log(pi[k]) + stats.gamma(shape_params[k], loc=0, scale=scale_params[k]).pdf(x[0])
for i in range(1, N):
    for k in range(K):
        omega[

