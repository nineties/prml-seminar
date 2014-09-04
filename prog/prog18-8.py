# -*- coding: utf-8 -*0
from numpy import *
from matplotlib.pyplot import *
import matplotlib.cm as cm
import scipy.stats as stats
from sklearn.mixture import GMM

# 混合ガウス分布クラスタリングの実装例

N = 150
x = concatenate([
    random.multivariate_normal([0, 0], eye(2), N/3),
    random.multivariate_normal([0, 5], eye(2), N/3),
    random.multivariate_normal([2, 3], eye(2), N/3),
    ])

K = 3
gmm = GMM(n_components = K)
gmm.fit(x)


clf()
xlim(-3, 5)
ylim(-3, 8)
scatter(x[:, 0], x[:, 1], c = gmm.predict(x), s=50)
X, Y = meshgrid(linspace(-3, 5), linspace(-3, 8))
Z = vectorize(lambda x,y: gmm.score_samples([[x,y]])[0][0])(X, Y)
pcolor(X, Y, Z, alpha=0.2)
savefig('fig18-8.png')
