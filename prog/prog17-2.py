# -*- coding: utf-8 -*0
from numpy import *
from scipy import linalg as LA
from matplotlib.pyplot import *
import matplotlib.cm as cm
from struct import *

def read_ubyte(fp):
    return unpack(">B", fp.read(1))[0]

infile = open("bayes-noise.ppm", "rb")
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

BETA = 3.0
ETA  = 2.0
ENTH = 0.0

# ピクセル (i,j) をフリップした場合のエネルギーの増加
def diffE(x, y, i, j):
    p = x[i, j]
    q = - p
    e = ENTH*(q-p)
    if i < W-1:
        e -= BETA * (q-p) *x[i+1, j]
    if i > 0:
        e -= BETA * (q-p)*x[i-1, j]
    if j < H-1:
        e -= BETA * (q-p)*x[i, j+1]
    if j > 0:
        e -= BETA * (q-p)*x[i, j-1]
    e -= ETA * (q-p) * y[i, j]
    return e

new_img = copy(img)
changed = True
while changed:
    print "."
    changed = False
    for j in range(H):
        for i in range(W):
            if diffE(new_img, img, i, j) < 0:
                new_img[i, j] = - new_img[i, j]
                changed = True

outfile = open("bayes-denoise2.ppm", "wb")
outfile.write("P6\n")
outfile.write("%d %d\n" % (W, H))
outfile.write("255\n")
for j in range(H):
    for i in range(W):
        value = new_img[i,j]
        if value == -1:
            outfile.write(array([0, 0, 0], dtype=uint8))
        else:
            outfile.write(array([255, 255, 255], dtype=uint8))
outfile.close()
