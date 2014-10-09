from numpy import *
from matplotlib.pyplot import *
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import LinearSVC

N = 50
x = random.uniform(0, 1, (N, 2))
t = x[:,0]**2 < x[:,1]

xlim(0, 1)
ylim(0, 1)
scatter(x[:, 0], x[:, 1], c = t, s=50, cmap=cm.cool)
savefig('prog23-1-0.png')

for M in [1, 10, 100, 1000, 10000]:
    classifier = BaggingClassifier(base_estimator=LinearSVC(),
            n_estimators=M)
    classifier.fit(x, t)

    X, Y = meshgrid(linspace(0, 1), linspace(0, 1))
    Z = classifier.predict(c_[X.ravel(), Y.ravel()])
    Z = Z.reshape(X.shape)

    clf()
    xlim(0, 1)
    ylim(0, 1)
    scatter(x[:, 0], x[:, 1], c = t, s=50, cmap=cm.cool)
    pcolor(X, Y, Z, alpha=0.5)
    savefig('prog23-1-%d.png' % M)
