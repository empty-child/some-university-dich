import cv2 as cv
import math
import numpy as np
from matplotlib import pyplot as plt

log_mask = np.array([[2, 2, 4, 2, 2],
                     [2, -1, -5, -1, 2],
                     [4, -5, -16, -5, 4],
                     [2, -1, -5, -1, 2],
                     [2, 2, 4, 2, 2]], "i")


def convolution(matr, img):
    out_img = np.copy(img)
    rows, columns = img.shape[:2]
    for i in range(columns)[2:columns-2]:
        for j in range(rows)[2:rows-2]:
            x = 0
            matrsum = [0, 0, 0]
            for i2 in [j-2, j-1, j, j+1, j+2]:
                y = 0
                for j2 in [i-2, i-1, i, i+1, i+2]:
                    matrsum[0] += img.item(j2, i2, 0)*matr[x][y]
                    matrsum[1] += img.item(j2, i2, 1) * matr[x][y]
                    matrsum[2] += img.item(j2, i2, 2) * matr[x][y]
                    y += 1
                x += 1
            out_img.itemset((j, i, 0), math.fabs(matrsum[0]))
            out_img.itemset((j, i, 1), math.fabs(matrsum[1]))
            out_img.itemset((j, i, 2), math.fabs(matrsum[2]))

            # out_img.itemset((j, i, 0), math.fabs((img.item(j-1, i-1, 0)*matr[0][0]+img.item(j-1, i, 0)*matr[0][1]+
            #                          img.item(j-1, i+1, 0)*matr[0][2]+img.item(j, i-1, 0)*matr[1][0]+
            #                          img.item(j, i, 0)*matr[1][1]+img.item(j, i+1, 0)*matr[1][2]+
            #                          img.item(j+1, i-1, 0)*matr[2][0]+img.item(j+1, i, 0)*matr[2][1]+
            #                          img.item(j+1, i+1, 0)*matr[2][2])))
            # out_img.itemset((j, i, 1), math.fabs((img.item(j-1, i-1, 1)*matr[0][0]+img.item(j-1, i, 1)*matr[0][1]+
            #                          img.item(j-1, i+1, 1)*matr[0][2]+img.item(j, i-1, 1)*matr[1][0]+
            #                          img.item(j, i, 1)*matr[1][1]+img.item(j, i+1, 1)*matr[1][2]+
            #                          img.item(j+1, i-1, 1)*matr[2][0]+img.item(j+1, i, 1)*matr[2][1]+
            #                          img.item(j+1, i+1, 1)*matr[2][2])))
            # out_img.itemset((j, i, 2), math.fabs((img.item(j-1, i-1, 2)*matr[0][0]+img.item(j-1, i, 2)*matr[0][1]+
            #                          img.item(j-1, i+1, 2)*matr[0][2]+img.item(j, i-1, 2)*matr[1][0]+
            #                          img.item(j, i, 2)*matr[1][1]+img.item(j, i+1, 2)*matr[1][2]+
            #                          img.item(j+1, i-1, 2)*matr[2][0]+img.item(j+1, i, 2)*matr[2][1]+
            #                          img.item(j+1, i+1, 2)*matr[2][2])))
    return out_img

img = cv.imread("3.jpg")
img = cv.resize(img,(555,555))
img2 = convolution(log_mask, img)
plt.subplot(1,2,1),plt.imshow(img)
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(img2)
plt.title('LoG'), plt.xticks([]), plt.yticks([])
plt.show()