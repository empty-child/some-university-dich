import cv2 as cv
import math
import numpy as np
from matplotlib import pyplot as plt

laplacian_matrix = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], "i")
sobel_matrix_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], "i")
sobel_matrix_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], "i")


def convolution(matr, img):
    out_img = np.copy(img)
    rows, columns = img.shape[:2]
    for i in range(columns)[1:columns-1]:
        for j in range(rows)[1:rows-1]:
            out_img.itemset((j, i, 0), math.fabs((img.item(j-1, i-1, 0)*matr[0][0]+img.item(j-1, i, 0)*matr[0][1]+
                                     img.item(j-1, i+1, 0)*matr[0][2]+img.item(j, i-1, 0)*matr[1][0]+
                                     img.item(j, i, 0)*matr[1][1]+img.item(j, i+1, 0)*matr[1][2]+
                                     img.item(j+1, i-1, 0)*matr[2][0]+img.item(j+1, i, 0)*matr[2][1]+
                                     img.item(j+1, i+1, 0)*matr[2][2])))
            out_img.itemset((j, i, 1), math.fabs((img.item(j-1, i-1, 1)*matr[0][0]+img.item(j-1, i, 1)*matr[0][1]+
                                     img.item(j-1, i+1, 1)*matr[0][2]+img.item(j, i-1, 1)*matr[1][0]+
                                     img.item(j, i, 1)*matr[1][1]+img.item(j, i+1, 1)*matr[1][2]+
                                     img.item(j+1, i-1, 1)*matr[2][0]+img.item(j+1, i, 1)*matr[2][1]+
                                     img.item(j+1, i+1, 1)*matr[2][2])))
            out_img.itemset((j, i, 2), math.fabs((img.item(j-1, i-1, 2)*matr[0][0]+img.item(j-1, i, 2)*matr[0][1]+
                                     img.item(j-1, i+1, 2)*matr[0][2]+img.item(j, i-1, 2)*matr[1][0]+
                                     img.item(j, i, 2)*matr[1][1]+img.item(j, i+1, 2)*matr[1][2]+
                                     img.item(j+1, i-1, 2)*matr[2][0]+img.item(j+1, i, 2)*matr[2][1]+
                                     img.item(j+1, i+1, 2)*matr[2][2])))
    return out_img


def log_trans(img):
    table = np.array([((math.log10(1+i/255))*850) for i in np.arange(0, 256)]).astype("uint8")
    return cv.LUT(img, table)


def pow_trans(img, gamma=1):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv.LUT(img, table)


img = cv.imread("11.jpg")
img2 = log_trans(img)
img3 = pow_trans(img,0.4)
laplacian = cv.Laplacian(img,cv.CV_8U, ksize=1)
sobelx = cv.Sobel(img,cv.CV_8U,1,0)
sobely = cv.Sobel(img,cv.CV_8U,0,1)
img4 = convolution(laplacian_matrix, img)
img5 = convolution(sobel_matrix_x, img)
img6 = convolution(sobel_matrix_y, img)

plt.subplot(3,3,1),plt.imshow(img)
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,2),plt.imshow(img2)
plt.title('Logarithmic transformation'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,3),plt.imshow(img3)
plt.title('Power transformation'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,4),plt.imshow(laplacian)
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,5),plt.imshow(sobelx)
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,6),plt.imshow(sobely)
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,7),plt.imshow(img4)
plt.title('Laplacian. Full realization.'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,8),plt.imshow(img5)
plt.title('Sobel X. Full realization.'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,9),plt.imshow(img6)
plt.title('Sobel Y. Full realization.'), plt.xticks([]), plt.yticks([])

plt.show()
