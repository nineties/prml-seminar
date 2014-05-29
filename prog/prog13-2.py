# -*- coding: utf-8 -*-
from numpy import *
from scipy import linalg as LA
from matplotlib.pyplot import *

#== Relevance vector machine による回帰 ==

MAX_ITER   = 1000
ITER_EPS   = 1.0e-2
KERN_SIGMA = 0.3
THRESHOLD  = 1.0e2 # これよりαが大きいならばw=0とする.

# 学習データ
random.seed(0)
N = 50
train_x = random.uniform(-1, 1, N)
train_t = random.normal(sin(2*pi*train_x), 0.5)

def kernel(x1, x2):
    return exp(-sum((x1-x2)**2)/(2*KERN_SIGMA**2))

# 計画行列
X = ones((N, N+1))
for i in range(N):
    for j in range(N):
        X[i, j] = kernel(train_x[i], train_x[j])

alpha = random.uniform(0, 1, N+1)
beta  = 1.0
for i in range(MAX_ITER):
    A = diag(alpha)
    SIGMA = LA.inv(A + beta * X.T.dot(X))
    m = beta * SIGMA.dot(X.T.dot(train_t))
    gamma = 1 - alpha*diagonal(SIGMA)
    new_alpha = gamma / m**2
    new_alpha[where(new_alpha > THRESHOLD)] = THRESHOLD # Overflow対策
    num   = LA.norm(train_t - X.dot(m))**2
    denom = N - sum(gamma)
    new_beta = denom/num

    if LA.norm(new_alpha-alpha)/LA.norm(new_alpha) < ITER_EPS and\
       abs(new_beta-beta)/abs(new_beta) < ITER_EPS: break
    
    alpha = new_alpha
    beta  = new_beta
print "count=",i

# w = m
A = diag(alpha)
SIGMA = LA.inv(A + beta * X.T.dot(X))
w = beta * SIGMA.dot(X.T.dot(train_t))

# 回帰方程式
def f(w, x):
    # 以下ではalpha[i]が大きな点を無視するだけに
    # していますが, 実際には学習完了後にそれらを
    # 消去してしまってよいです.

    v = 0
    for i in range(N):
        if alpha[i] >= THRESHOLD: continue
        v += w[i]*kernel(train_x[i], x)
    v += w[N]
    return v

# 標準偏差
def std(w, x):
    s = 1/beta
    for i in range(N):
        for j in range(N):
            s += SIGMA[i,j]*kernel(train_x[i],x)*kernel(train_x[j],x)
    return sqrt(s)

x = linspace(-1, 1, 100)
y = vectorize(lambda x: f(w, x))(x)
s = vectorize(lambda x: std(w, x))(x)

xlim(-1, 1)
ylim(-2, 2)
count = 0
for i in range(N):
    color = "blue"
    if alpha[i] < THRESHOLD:
        count += 1
        color = "red"
    plot(train_x[i], train_t[i], "o", color=color)
plot(x, y)
plot(x, y+s)
plot(x, y-s)
show()

print "%d/%d" % (count, N)
