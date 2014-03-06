# -*- coding: utf-8 -*-
from numpy import *
from scipy import stats
from scipy import linalg as LA
from matplotlib.pyplot import *
import matplotlib.cm as cm

xmin = 0
xmax = 1
ymin = -1.5
ymax = 1.5

# ガウス基底の数
NGAUSS = 5
GAUSS_S = 0.1 # ガウス基底の分散パラメータ
GAUSS_MU = linspace(xmin, xmax, NGAUSS) # ガウス基底の位置パラメータ

# aの事前分布のパラメータ
a0  = zeros(NGAUSS)
s02 = 10

# σ^2の事前分布のパラメータ
alpha0 = 1.0
beta0  = 1.0

# 学習データ
data = loadtxt("prog4-5.dat")
x = data[:,0]
y = data[:,1]
n = len(x)

# 繰り返し使う行列・ベクトルを作っておく
def psi(x):
    return array([exp(-(x - m)**2/(2*GAUSS_S**2)) for m in GAUSS_MU])
X = array(psi(x)).T
XtX = X.T.dot(X)
Xty = X.T.dot(y)

# π(a|σ^2,D)
def p_a(s2):
    Sigma  = LA.inv(XtX/s2 + 1/s02)
    a_dash = Sigma.dot(Xty/s2 + a0/s02)
    return random.multivariate_normal(a_dash, Sigma)

# π(σ^2|a,D)
def p_s2(a):
    alpha = alpha0 + n
    beta  = beta0  + LA.norm(y-X.dot(a))**2
    # X ～ IG(a,b)  <=>  1/X ～ Ga(a,1/b)
    return 1/random.gamma(alpha/2, 2/beta)

# ギブスサンプリング
def sampler(BURNIN):
    a  = zeros(NGAUSS)   # 係数の初期値
    s2 = 1.0        # 分散の初期値
    for i in range(BURNIN):
        a  = p_a(s2)
        s2 = p_s2(a)
    while True:
        a  = p_a(s2)
        s2 = p_s2(a)
        yield a, s2

sim = sampler(1000)

NSAMPLE = 10000
a_samples = zeros((NSAMPLE,NGAUSS))
s2_samples = zeros(NSAMPLE)
for i in range(NSAMPLE):
    a_samples[i], s2_samples[i] = sim.next()

# MAP 推定値
a_MAP  = [ average(a_samples[:,i]) for i in range(NGAUSS)]
s2_MAP = average(s2_samples)

xlim(xmin, xmax)
ylim(ymin, ymax)
t = linspace(xmin, xmax, 100)
scatter(x, y)
plot(t, sin(2*pi*t))
plot(t, psi(t).T.dot(a_MAP))
show()

# 予測分布
def predictive(sim, x):
    a, s2 = sim.next()
    return random.normal(psi(x).T.dot(a), sqrt(s2))

xlim(xmin, xmax)
ylim(ymin, ymax)
xs = linspace(xmin, xmax, 100)
means = []
ys_high = []
ys_low  = []
for t in xs:
    NGAUSS = 100
    mean = psi(t).T.dot(a_MAP)
    s = sqrt(average([(mean - predictive(sim, t))**2 for i in range(NGAUSS)]))
    means.append(mean)
    ys_high.append(mean + s)
    ys_low.append(mean - s)

scatter(x, y)
plot(xs, means)
plot(xs, ys_high)
plot(xs, ys_low)
show()
