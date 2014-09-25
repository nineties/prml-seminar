# -*- coding: utf-8 -*-
from numpy import *
from matplotlib.pyplot import *
import scipy.stats as stats
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from zipfile import ZipFile

# UCI Machine Learning Repositoryの
# http://archive.ics.uci.edu/ml/datasets/Activity+Recognition+from+Single+Chest-Mounted+Accelerometer
# をダウンロードし同じディレクトリに置く
NAME = 'Activity Recognition from Single Chest-Mounted Accelerometer'
ar = ZipFile(open(NAME + '.zip'))
df = pd.read_csv(ar.open(NAME + '/1.csv'), header=None)[[1,2,3,4]]
df.columns = ['x', 'y', 'z', 'label']

N = len(df)
colors = ['blue', 'red', 'green', 'yellow', 'cyan', 'magenta', 'purple']

xlim(0, N)
plot(df['x'], label='x')
plot(df['y'], label='y')
plot(df['z'], label='z')
legend()

i = 0
while i < N:
    label = df['label'][i]
    print label
    left = i
    while i+1 < N and df['label'][i+1] == label:
        i += 1
    right = i
    if left==right: break
    axvspan(left, right, facecolor=colors[label-1], alpha=0.5)
    i += 1

show()
