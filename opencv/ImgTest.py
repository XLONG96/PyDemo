'''
Created on 2018年4月18日

@author: Administrator
'''
import cv2
import numpy as np
#from matplotlib import pyplot as plt
import sys,time

def equ(img):
    
    equ = cv2.equalizeHist(img)
    res = np.hstack((img, equ))
    
    #cv2.imshow('image', res)
    return res

def catchBlack(img):
    lower_black = np.array([110,50,50])
    upper_black = np.array([130,255,255])
    
    #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(img, lower_black, upper_black)
    
    res = cv2.bitwise_and(img, img, mask=mask)
    
    cv2.imshow('image', img)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)

def erodeImg(img):
    kernel = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(img, kernel);
    return eroded
    #cv2.imshow('image', eroded)
    #cv2.imwrite('D:/Downloads/temp.jpg', eroded)

def countGray(img):
    count = []

    for i in range(0, 256):
        count.append(0)
    
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            #print(img[i,j])
            #time.sleep(3)
            count[img[i, j]] += 1
            
    for i in range(0, len(count)):
        if i % 10 == 0:
            print()
        print(count[i], end=" ")  

def pipeLine(img):
    a = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
    a[:,:] = img[:,:,1]
    
    return a

def delWhite(img):
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i, j] > 180:
                img[i, j] = 0
            if img[i, j] == 0:
                img[i, j] = 255
    return img
        
def thresholdImg(img):
    img = cv2.medianBlur(img, 5)
    ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
                                cv2.THRESH_BINARY, 4, 10)
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                cv2.THRESH_BINARY, 3, 2)
    
    #cv2.imshow('img', img)
    #cv2.imshow('th1', th1)
    cv2.imshow('th2', th2)
    #cv2.imshow('th3', th3)

def otsuImg(img):
    #blur = cv2.GaussianBlur(img, (1,1), 0)
    ret, th3 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    cv2.imshow('th3', th3)
    
img = cv2.imread('D:/Downloads/all-item/temp.jpg', 1)

print(img.shape)
#cv2.imwrite('D:/Downloads/temp.jpg', img)

#f = np.fft.fft2(img)
#fshift = np.fft.fftshift(f)

#img = equ(img)

#catchBlack(img)


#img = pipeLine(img)

'''
img = delWhite(img)
img = cv2.bitwise_not(img)
img = erodeImg(img)
countGray(img)
#thresholdImg(img)
#otsuImg(img)
'''
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#img = np.zeros((50,50, 1), np.uint8)

#cv2.imwrite('D:/Downloads/temp.jpg', img)
