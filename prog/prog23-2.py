from numpy import *
from matplotlib.pyplot import *
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression

N = 100
x = random.uniform(0, 1, N)
t = random.normal(sin(x*pi/2), 0.3)

N_TRAIN = 50

x_train = x[:N_TRAIN]
t_train = t[:N_TRAIN]
x_test  = x[N_TRAIN:]
t_test  = t[N_TRAIN:]

scatter(x_train, t_train, label='trainig', color='blue')
scatter(x_test, t_test, label='test', color='red')
legend()
savefig('prog23-2-1.png')

err = []
for M in arange(1, 50):
    m = 0
    for i in range(1000):
        reg = BaggingRegressor(base_estimator=LinearRegression(), n_estimators=M)
        reg.fit(c_[x_train], t_train)
        m += average((reg.predict(c_[x_test]) - t_test)**2)
    err.append( m/1000 )

clf()
title('mean squared error')
ylabel('mse')
xlabel('number of weak-learner (M)')
plot(err)
savefig('prog23-2-2.png')
