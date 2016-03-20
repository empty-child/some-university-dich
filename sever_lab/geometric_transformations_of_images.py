import cv2 as cv
import math
import numpy as np
from matplotlib import pyplot as plt


# def log_trans():
#     for i in range(columns):
#         for j in range(rows):
#             # img.itemset((j, i, 0), math.log(1+img.item(j, i, 0)))
#             # img.itemset((j, i, 1), math.log(1+img.item(j, i, 1)))
#             img.itemset((j, i, 2), math.log(1+img.item(j, i, 2)))
#             # print(img.item(j, i, 1))

def log_trans(img):
    table = np.array([((math.log(1+i/255))*255) for i in np.arange(0, 256)]).astype("uint8")
    # print(table)
    return cv.LUT(img, table)


def pow_trans(img, gamma=1):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv.LUT(img, table)

img = cv.imread("1.jpg")
rows, columns = img.shape[:2]
# img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
img2 = log_trans(img)
img3 = pow_trans(img,0.4)
# img = cv.cvtColor(img, cv.COLOR_HSV2RGB)
laplacian = cv.Laplacian(img,cv.CV_8U)
sobelx = cv.Sobel(img,cv.CV_8U,1,0)
sobely = cv.Sobel(img,cv.CV_8U,0,1)
plt.subplot(2,3,1),plt.imshow(img)
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2),plt.imshow(img2)
plt.title('Logarithmic transformation'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3),plt.imshow(img3)
plt.title('Power transformation'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,4),plt.imshow(laplacian)
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,5),plt.imshow(sobelx)
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,6),plt.imshow(sobely)
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()
