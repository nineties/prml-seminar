from numpy import *
from matplotlib.pyplot import *
from sklearn.tree import DecisionTreeRegressor

N = 100
x = random.uniform(0, 1, N)
t = zeros(N)
t[x < 0.5] = random.normal(0.5*x[x < 0.5], 0.1)
t[x >= 0.5] = random.normal(1.25-0.5*x[x >= 0.5], 0.1)

scatter(x, t)
show()

reg = DecisionTreeRegressor(max_depth=10)
reg.fit(c_[x], t)

plt_x = linspace(0, 1)
clf()
plot(plt_x, reg.predict(c_[plt_x]))
scatter(x, t)
show()
