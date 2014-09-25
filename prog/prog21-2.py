# -*- coding: utf-8 -*-
from numpy import *
from matplotlib.pyplot import *
import re

# UJI/ に
# http://archive.ics.uci.edu/ml/datasets/UJI+Pen+Characters
# からダウンロードしたファイルを置く

def read_traj_data(person, char):
    # 一人あたり2データ
    traj = [[], []]
    f = open('UJI/UJIpenchars-w%02d' % person)
    # .SEGMENT CHARACTER ... という行を見つける
    pat = re.compile('.SEGMENT CHARACTER \d+ \? "%s"' % char)
    cnt = 0
    while True:
        line = f.readline()
        if not line: break
        result = pat.search(line)
        if result:
            f.readline()
            f.readline()
            f.readline()
            while True:
                line = f.readline().strip()
                if line == '.PEN_UP':
                    break
                traj[cnt].append( map(float, line.split()) )
            traj[cnt] = array(traj[cnt])
            cnt += 1
    f.close()
    return traj

# 11人分布の'a'の手書きデータを読み出す
data = []
for person in range(1, 11+1):
    data += read_traj_data(person, 'a')

print data[0]

# 出力
for i in range(22):
    x = data[i][:, 0]
    y = data[i][:, 1]

    clf()
    gca().invert_yaxis()
    scatter(x, y, s=50)
    plot(x, y)
    savefig('UJI-%d.png' % i)
