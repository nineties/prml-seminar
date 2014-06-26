# -*- coding: utf-8 -*0
from numpy import *
from scipy import linalg as LA
from matplotlib.pyplot import *
import matplotlib.cm as cm
from struct import *

def read_ubyte(fp):
    return unpack(">B", fp.read(1))[0]

infile = open("bayes.ppm", "rb")
infile.readline() # シグネチャ. チェック省略
W, H = map(int, infile.readline().split(" ")) # 幅, 高さ
infile.readline()  # 255. チェック省略

img = zeros((W, H), dtype=int8)
for j in range(H):
    for i in range(W):
        r = read_ubyte(infile)
        read_ubyte(infile)
        read_ubyte(infile)
        if r == 255:
            img[i, j] = 1
        else:
            img[i, j] = -1

outfile = open("bayes-original.ppm", "wb")
outfile.write("P6\n")
outfile.write("%d %d\n" % (W, H))
outfile.write("255\n")
for j in range(H):
    for i in range(W):
        if img[i,j] == -1:
            outfile.write(array([0, 0, 0], dtype=uint8))
        else:
            outfile.write(array([255, 255, 255], dtype=uint8))
outfile.close()

P_FLIP = 0.1
outfile = open("bayes-noise.ppm", "wb")
outfile.write("P6\n")
outfile.write("%d %d\n" % (W, H))
outfile.write("255\n")
for j in range(H):
    for i in range(W):
        value = img[i,j]
        if random.uniform() < P_FLIP:
            value = -value
        if value == -1:
            outfile.write(array([0, 0, 0], dtype=uint8))
        else:
            outfile.write(array([255, 255, 255], dtype=uint8))
outfile.close()
