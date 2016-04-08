import cv2 as cv
import math
import numpy as np
from matplotlib import pyplot as plt

log_mask = np.array([[0,0,3,2,2,2,3,0,0],
                      [0,2,3,5,5,5,3,2,0],
                      [3,3,5,3,0,3,5,3,3],
                      [2,5,3,-12,-23,-12,3,5,2],
                      [2,5,0,-23,-40,-23,0,5,2],
                      [2,5,3,-12,-23,-12,3,5,2],
                      [3,3,5,3,0,3,5,3,3],
                      [0,2,3,5,5,5,3,2,0],
                      [0,0,3,2,2,2,3,0,0]], "i")


def convolution_from_grayscale(matr, img):
    koef = len(matr)//2
    out_img = np.copy(img)
    rows, columns = img.shape[:2]
    ar = np.zeros([rows, columns], dtype="i2")
    for i in range(columns)[koef:columns-koef]:
        for j in range(rows)[koef:rows-koef]:
            x = 0
            matrsum = 0
            for j2 in range(-koef, koef+1):
                y = 0
                for i2 in range(-koef, koef+1):
                    matrsum += img.item(j+j2, i+i2) * matr[x][y]
                    y += 1
                x += 1
            ar[j-koef][i-koef] = matrsum
            out_img.itemset((j, i), 0) if ar[j][i] > 0 else out_img.itemset((j, i), 255)
    return out_img

img = cv.imread("1.jpg")
img2 = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
img2 = convolution_from_grayscale(log_mask, img2)
# cv.imshow("img", img2)
# cv.waitKey()
plt.subplot(1,2,1),plt.imshow(img)
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(img2)
plt.title('Marr-Hildreth Algorithm'), plt.xticks([]), plt.yticks([])
plt.show()