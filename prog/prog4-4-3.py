# -*- coding: utf-8 -*-
from numpy import *
from scipy import stats
from scipy import linalg as LA
from matplotlib.pyplot import *
import matplotlib.cm as cm

# aの事前分布のパラメータ
a0  = array([0,0])
s02 = 1.0e2 ** 2

# σ^2の事前分布のパラメータ
alpha0 = 1.0e-2
beta0  = 1.0e2

# 学習データ
data = loadtxt("prog4-4.dat")
x = data[:,0]
y = data[:,1]
n = len(x)

# 繰り返し使う行列・ベクトルを作っておく
X = array([x, ones(n)]).T
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

# ギブスサンプラー
def sampler(BURNIN):
    a  = array([0, 0])  # 係数の初期値
    s2 = 1.0            # 分散の初期値
    for i in range(BURNIN):
        a  = p_a(s2)
        s2 = p_s2(a)
    while True:
        a  = p_a(s2)
        s2 = p_s2(a)
        yield a[0], a[1], s2

sim = sampler(1000)

NSAMPLE = 10000
a_samples = zeros(NSAMPLE)
b_samples = zeros(NSAMPLE)
s2_samples = zeros(NSAMPLE)
for i in range(NSAMPLE):
    a_samples[i], b_samples[i], s2_samples[i] = sim.next()

# 事後分布の図示
def posterior(name, samples):
    NBINS = 100
    title(u"posterior distribution of %s (N=%d, mean=%.2f, std=%.2f)"
            % (name, NSAMPLE, average(samples), std(samples)))
    hist(samples, bins=NBINS, normed=True)
    show()

posterior("a", a_samples)
posterior("b", b_samples)
posterior(u"σ^2", s2_samples)

#MAP 推定値
a_MAP  = average(a_samples)
b_MAP  = average(b_samples)
s2_MAP = average(s2_samples)

# 回帰直線の表示
xmin = 0;  xmax = 1
ymin = -1; ymax = 5
xlim(xmin, xmax)
ylim(ymin, ymax)
t = linspace(0, 1, 100)
regress_line = a_MAP*t + b_MAP
xlim(0, 1)
scatter(x, y)
plot(t, regress_line)
fill_between(t, regress_line+sqrt(s2_MAP), regress_line-sqrt(s2_MAP),
        alpha=0.2)
title(u"regression line")
show()

# 予測分布
def predictive(sim, x):
    a,b,s2 = sim.next()
    return random.normal(a*x+b, sqrt(s2))

xlim(xmin, xmax)
ylim(ymin, ymax)
xs = linspace(xmin, xmax, 100)
means = []
ys_high = []
ys_low  = []
for t in xs:
    M = 100
    mean = a_MAP*t + b_MAP
    s = sqrt(average([(mean - predictive(sim, t))**2 for i in range(M)]))
    means.append(mean)
    ys_high.append(mean + s)
    ys_low.append(mean - s)

scatter(x, y)
plot(xs, means)
plot(xs, ys_high)
plot(xs, ys_low)
show()
