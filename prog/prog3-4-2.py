# -*- coding: utf-8 -*-
from numpy import *
from scipy import stats
from matplotlib.pyplot import *

# 真の値(台形近似で. N=100だと4桁は合うはず.)
def answer():
    N = 100
    x = linspace(0, 2, N+1)
    fx = stats.norm.pdf(x)
    return 0.5 - (sum(2*fx)-fx[0]-fx[-1])*2/(2*N)
answer = answer()

def integrate(n):
    x = random.randn(n)
    return float(count_nonzero(x >= 2))/n

M = 100

# 近似値の絶対誤差をM個の平均をプロット
n = arange(1000, 50001, 1000)
mae = vectorize(
        lambda n: average([abs(answer - integrate(n)) for i in range(M)])
        )(n)
plot(n, mae)
plot(n, [0.001]*len(n))
xlabel('number of samples (N)')
ylabel('mean absolute error (MAE)')
show()
