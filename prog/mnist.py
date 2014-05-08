# -*- coding: utf-8 -*0
from numpy import *
from scipy import linalg as LA
from matplotlib.pyplot import *
import matplotlib.cm as cm
from struct import *

# 以下はファイルから読むべきだけどこのサンプルでは省略
N      = 60000 # 学習データ数
WIDTH  = 28    # 画像の幅
HEIGHT = 28    # 画像の高さ
N_TEST = 10000 # テスト用データ数

#=== 画像とラベルの読み書き ===
def read_ubyte(fp):
    return unpack(">B", fp.read(1))[0]

def read_uint(fp):
    return unpack(">I", fp.read(4))[0]

def read_header(imgfile, lblfile):
    read_uint(img)       # マジックナンバー. チェック省略
    N = read_uint(img)   # データ数
    h = read_uint(img)   # height
    w = read_uint(img)   # width

# 画像データ(uint8配列)・ラベルのペアを返すジェネレータを生成
def make_dataset(imgfile, lblfile, n):
    img = open(imgfile, "rb")
    lbl = open(lblfile, "rb")
    img.read(4*4) # マジックナンバー,データ数,高さ,幅をスキップ
    lbl.read(4*2) # マジックナンバー,データ数をスキップ

    for i in range(n):
        data  = fromstring(img.read(HEIGHT*WIDTH), dtype = uint8)/255.0
        label = read_ubyte(lbl)
        yield data, label

    img.close()
    lbl.close()

# PGM画像生成
def write_image(fname, data, glid=False):
    data = uint8(data*255).reshape((HEIGHT, WIDTH))

    if glid:
        data[HEIGHT/2,:] = 255
        data[:,WIDTH/2]  = 255

    fp = open(fname, "wb")
    fp.write("P5\n")
    fp.write("%d %d\n" %(WIDTH, HEIGHT))
    fp.write("255\n")
    fp.write(data)
    fp.close()

#=== ニューラルネットワーク ===
D = WIDTH*HEIGHT # 学習データの次元
M = 20           # 隠れ層の数
K = 10           # 出力層の数 0-9

ALPHA = 0.05

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
    return delta2.dot((1- tanh(a1)**2)*w2[:,0:M]) # 隠れ層の誤差

# 偏微分係数の計算
def diffcoef(x, a1, a2, w1, w2, delta2):
    delta1 = backprop(a1, a2, w1, w2, delta2)
    diff1 = outer(delta1, append(x, 1))
    diff2 = outer(delta2, append(tanh(a1), 1))
    return (diff1, diff2)

#=== ソフトマックス関数 ===

# a: 出力層の入力
def softmax(a):
    expa = exp(a)
    return expa/sum(expa)

#=== 確率的勾配降下法 ===
# count番目のデータ(x, t)を学習し重みw1, w2を更新
def study(count, w1, w2, x, t):
    a1, a2 = forward(x, w1, w2)
    y = softmax(a2)
    d1, d2 = diffcoef(x, a1, a2, w1, w2, y-t)
    w1 -= ALPHA*d1
    w2 -= ALPHA*d2

#=== 識別 ===
def classify(x, w1, w2):
    a1, a2 = forward(x, w1, w2)
    return argmax(softmax(a2))

#=== 実験 ===

# 学習用データの構築
dataset = make_dataset("train-images-idx3-ubyte", "train-labels-idx1-ubyte", N)
x = zeros((N, D))
t = zeros((N, K))

print "Studying %d data"%N
sys.stdout.flush()

w1 = random.uniform(-1, 1, (M, D+1))
w2 = random.uniform(-1, 1, (K, M+1))

i = 1
for img, lbl in dataset:
    t = zeros(K)
    t[lbl] = 1
    study(i, w1, w2, img, t)
    if i <= 20:
        write_image("minst-img%d.pgm"%i, img, glid=False)
    i += 1
print "Finished."
sys.stdout.flush()

print "Testing %d data."%N_TEST
dataset = make_dataset("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", N_TEST)
correct = 0
for img, lbl in dataset:
    t = classify(img, w1, w2)
    if lbl == t:
        correct += 1
print "correct answer rate = %d/%d" % (correct, N_TEST)
