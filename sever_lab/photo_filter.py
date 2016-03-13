import cv2 as cv
import random
import time


def negative():
    t1 = time.clock()
    for i in range(columns):
        for j in range(rows):
            img.itemset((j, i, 0), 255-img.item(j, i, 0))
            img.itemset((j, i, 1), 255-img.item(j, i, 1))
            img.itemset((j, i, 2), 255-img.item(j, i, 2))
    t2 = time.clock()
    return t2-t1


def grayscale():
    t1 = time.clock()
    for i in range(columns):
        for j in range(rows):
            avg = (img.item(j, i, 0) + img.item(j, i, 1) + img.item(j, i, 2))/3
            img.itemset((j, i, 0), avg)
            img.itemset((j, i, 1), avg)
            img.itemset((j, i, 2), avg)
    t2 = time.clock()
    return t2-t1


def sepia():
    t1 = time.clock()
    for i in range(columns):
        for j in range(rows):
            avg = (img.item(j, i, 0) + img.item(j, i, 1) + img.item(j, i, 2))/3
            img.itemset((j, i, 0), avg*0.41)
            img.itemset((j, i, 1), avg*0.71)
            img.itemset((j, i, 2), avg)
    t2 = time.clock()
    return t2-t1


def noise():
    t1 = time.clock()
    for i in range(columns):
            for j in range(rows):
                if j%2 == 0:
                    img.itemset((j, i, 0), random.randint(0, 255))
                    img.itemset((j, i, 1), random.randint(0, 255))
                    img.itemset((j, i, 2), random.randint(0, 255))
    t2 = time.clock()
    return t2-t1


def transparency():
    print()


img = cv.imread("1.jpg")
print("Выберите операцию: ")
ent = int(input())
rows, columns = img.shape[:2]
if ent == 1:
    t1 = negative()
    print("t1 = " + str(t1))
elif ent == 2:
    t2 = grayscale()
    print("t2 = " + str(t2))
elif ent == 3:
    t3 = sepia()
    print("t3 = " + str(t3))
elif ent == 4:
    t4 = noise()
    print("t4 = " + str(t4))
cv.imshow("img", img)
cv.waitKey()