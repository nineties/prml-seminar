from numpy import *
from matplotlib.pyplot import *
from sklearn.ensemble import BaggingClassifier

N = 300
x = random.uniform(-2, 2, (N, 2))
t = sin(x[:, 0]*pi) < x[:, 1]

xlim(-2, 2)
ylim(-2, 2)
scatter(x[:, 0], x[:, 1], c = t, s=50, cmap=cm.cool)
savefig('prog23-1-0.png')

for m in [1, 2, 3, 5, 10, 100, 1000]:
    classifier = BaggingClassifier(n_estimators=m)
    classifier.fit(x, t)

    X, Y = meshgrid(linspace(-2, 2), linspace(-2, 2))
    Z = classifier.predict(c_[X.ravel(), Y.ravel()])
    Z = Z.reshape(X.shape)

    clf()
    xlim(-2, 2)
    ylim(-2, 2)
    title('M = %d' % m)
    scatter(x[:, 0], x[:, 1], c = t, s=50, cmap=cm.cool)
    pcolor(X, Y, Z, alpha=0.5)
    savefig('prog23-1-%d.png' % m)
