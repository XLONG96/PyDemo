'''
Created on 2018年4月25日

@author: Administrator
'''
import cv2
import numpy as np
from math import ceil 

def balance(img):
    blur = cv2.GaussianBlur(img, (31,31), 0)
    return blur

# 对背景暗区与背景亮区的对比度进行补偿
def minusBk(A, B):
    F = 255
    ret = A
    [m, n] = A.shape
    for i in range(0, m):
        for j in range(0, n):
            k = setK(B[i, j])
            if B[i, j] > A[i, j]:
                ret[i, j] = F - k * (B[i, j] - A[i, j])
                if ret[i, j] < 0.75 * F:
                    ret[i, j] = 0.75 * F
            else:
                ret[i, j] = F
    return ret

def setK(e):
    if e < 20:
        k = 2.5
    elif e >= 20 and e <= 100:
        k = 1 + ((2.5 - 1) * (100 - e)) / 80
    elif e > 100 and e < 200:
        k = 1
    else:
        k = 1 + (e - 220) / 35
    return k

def thresholdImg(img):
    #img = cv2.medianBlur(img, 5)
    ret, th1 = cv2.threshold(img, 253, 255, cv2.THRESH_BINARY)
    ret, th4 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
                                cv2.THRESH_BINARY, 63, 3)
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                cv2.THRESH_BINARY, 3, 2)
    
    return th2
 
img = cv2.imread('D:/Downloads/all-item/16.jpg', 0)

print(img.shape[0], img.shape[1])

img = cv2.resize(img, (ceil(img.shape[1]/1), ceil(img.shape[0]/1)), interpolation=cv2.INTER_AREA)
#cv2.imshow('a', balance(img))

img = minusBk(img, balance(img))

cv2.imshow('b', img)

img = thresholdImg(img)

#cv2.imshow('b', thresholdImg(img))

cv2.imshow('img', img)

cv2.waitKey(0)
cv2.destroyAllWindows()



