# -*- coding: utf-8 -*-
from numpy import *
from scipy import linalg as LA
from matplotlib.pyplot import *

N = 30
xmin = -1; xmax = 1
ymin = -1; ymax = 1

#=== 学習データ ===
train_x = zeros((N,2))
i = 0
while True:
    x = random.uniform(xmin, xmax)
    y = random.uniform(ymin, ymax)

    # フィッシャーの判別法でないと上手くいかないような例を
    # 人工的に作る為のチェック
    if (y > x and x < -0.7) or (y < x and x > 0.7):
        continue

    train_x[i] = [x, y]
    i += 1
    if i == N: break

train_t = train_x[:,0] > train_x[:,1]

mu1 = average( train_x[train_t], axis=0 )
mu2 = average( train_x[logical_not(train_t)], axis=0 )


#=== フィッシャーの線形判別 ===
Sw = (train_x[train_t] - mu1).T.dot(train_x[train_t] - mu1) +\
     (train_x[logical_not(train_t)] - mu2).T.dot(train_x[logical_not(train_t)] - mu2)
w = LA.solve(Sw, mu1-mu2)

# wと直交する方向に射影
e = array([w[1], -w[0]])
e = e/LA.norm(e)

# Swを考慮しない場合
bad_e = array([(mu1-mu2)[1], -(mu1-mu2)[0]])
bad_e = bad_e/LA.norm(bad_e)

xlim(xmin, xmax)
ylim(ymin, ymax)
scatter(train_x[:,0], train_x[:,1], c=train_t, s=50, cmap=cm.cool)
plot(mu1[0], mu1[1], "bo", color="black")
plot(mu2[0], mu2[1], "bo", color="black")
arrow(0, 0, e[0], e[1], color="red", width=0.01)
arrow(0, 0, bad_e[0], bad_e[1], color="blue", width=0.01)
title("Fisher's linear discriminant")
savefig("fig6-4.png")
