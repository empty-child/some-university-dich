import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

log_mask = np.array([[0, 0, 3, 2, 2, 2, 3, 0, 0],
                     [0, 2, 3, 5, 5, 5, 3, 2, 0],
                     [3, 3, 5, 3, 0, 3, 5, 3, 3],
                     [2, 5, 3, -12, -23, -12, 3, 5, 2],
                     [2, 5, 0, -23, -40, -23, 0, 5, 2],
                     [2, 5, 3, -12, -23, -12, 3, 5, 2],
                     [3, 3, 5, 3, 0, 3, 5, 3, 3],
                     [0, 2, 3, 5, 5, 5, 3, 2, 0],
                     [0, 0, 3, 2, 2, 2, 3, 0, 0]], "i")

graycolors = [25, 51, 76, 102, 128, 166, 191, 229]


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
            out_img.itemset((j, i), 0) if ar[j-koef][i-koef] > 0 else out_img.itemset((j, i), 255)
    return out_img


def floodfilling(img):
    cnt = 0
    out_img = np.copy(img)
    rows, columns = out_img.shape[:2]
    for x in range(rows)[4:rows-4]:
        for y in range(columns)[4:columns-4]:
            if out_img[x,y] == 0:
                if cnt < len(graycolors)-1:
                    cnt += 1
                else:
                    cnt = 0
                out_img = FloodFill(out_img, [x, y], 0, graycolors[cnt])
    return out_img


def FloodFill( img, initNode, targetColor, replaceColor ):
    # initNode - координаты начала работы алгоритма в виде [x, y]
    xsize, ysize = img.shape[:2]
    Q = []
    if img[ initNode[0], initNode[1] ] != targetColor:
        return img
    Q.append( initNode )
    while Q != []:
        node = Q.pop(0)
        if img[ node[0], node[1] ] == targetColor:
            W = list( node )
        if node[0] + 1 < xsize:
            E = list( [ node[0] + 1, node[1] ] )
        else:
            E = list( node )
        while img[ W[0], W[1] ] == targetColor:
            img[ W[0], W[1] ] = replaceColor
            if W[1] + 1 < ysize:
                if img[ W[0], W[1] + 1 ] == targetColor:
                    Q.append( [ W[0], W[1] + 1 ] )
            if W[1] - 1 >= 0:
                if img[ W[0], W[1] - 1 ] == targetColor:
                    Q.append( [ W[0], W[1] - 1 ] )
            if W[0] - 1 >= 0:
                W[0] = W[0] - 1
            else:
                break
        while img[ E[0], E[1] ] == targetColor:
            img[ E[0], E[1] ] = replaceColor
            if E[1] + 1 < ysize:
                if img[ E[0], E[1] + 1 ] == targetColor:
                    Q.append( [ E[0], E[1] + 1 ] )
            if E[1] - 1 >= 0:
                if img[ E[0], E[1] - 1 ] == targetColor:
                    Q.append( [ E[0], E[1] -1 ] )
            if E[0] + 1 < xsize:
                E[0] = E[0] + 1
            else:
                break
    return img



img = cv.imread("1.jpg")
rows, columns = img.shape[:2]
img2 = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
img2 = convolution_from_grayscale(log_mask, img2)
img3 = floodfilling(img2)
cv.imshow("img", img3)
cv.waitKey()
# plt.subplot(1,3,1),plt.imshow(img)
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(1,3,2),plt.imshow(img2)
# plt.title('Marr-Hildreth Algorithm'), plt.xticks([]), plt.yticks([])
# plt.subplot(1,3,3),plt.imshow(img3)
# plt.title('Segmentation'), plt.xticks([]), plt.yticks([])
# plt.show()