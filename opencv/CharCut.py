'''
Created on 2018年4月25日

@author: Administrator
'''

import cv2
import numpy as np

def delWhite(img):
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i, j] > 180:
                img[i, j] = 0
            if img[i, j] == 0:
                img[i, j] = 255
    return img

def getTextProjection(img, pos, mode):
    if mode is 1:
        for i in range(0, img.shape[0] ):
            for j in range(0, img.shape[1]):
                if img[i, j] >= 245:
                    pos[i] += 1
    elif mode is 2:
        for i in range(0, img.shape[1]):
            for j in range(0, img.shape[0]):
                if img[j, i] >= 245:
                    pos[i] += 1

def dramProjection(pos, mode):
    if mode == 1:
        width = max(pos)
        height = len(pos)
        print(width, height)
        project = np.zeros((height, width), 'uint8')
        for i in range(0, project.shape[0]):
            for j in range(0, pos[i]):
                project[i, j] = 255
        
        for i in range(0, project.shape[0]):
            for j in range(0, project.shape[1]):
                #print(project[i, j])
                i += 2
        #cv2.imshow('img', project)
    elif mode == 2:
        height = max(pos)
        width = len(pos)
        project = np.zeros((height, width), 'uint8')
        for i in range(0, project.shape[1]):
            for j in range(height - pos[i], height):
                project[j, i] = 255
                
        #cv2.imshow('img', project)
    
    #cv2.waitKey(0)  

def getPeekRange(pos, peekRange):
    min_thresh= 2
    min_range = 13
    
    begin = 0
    end = 0
    for i in range(0, len(pos)):
        if pos[i] > min_thresh and begin is 0:
            begin = i
        elif pos[i] > min_thresh and begin != 0:
            continue
        elif pos[i] < min_thresh and begin != 0:
            end = i
            if end - begin >= min_range:
                peekRange.append([begin, end])
                begin = 0
                end = 0

def cutRow(img, peek):
    rowimg = img[peek[0]:peek[1], 0:img.shape[1]]
    return rowimg

def cutCol(rowimg):
    pos = [0] * img.shape[1]
    mode = 2
    
    getTextProjection(rowimg, pos, mode)         
    dramProjection(pos, mode)             
    
    peekRange = []
    getPeekRange(pos, peekRange)  
    
    return peekRange

def erodeImg(img):
    kernel = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(img, kernel);
    return eroded   
        
def pinRectangle(peek, peek2, img):
    for j in peek2:
        lef = (j[0], peek[0])
        rigt = (j[1], peek[1])
        cv2.rectangle(img, lef, rigt, (255, 0, 0), 1)
          
pos = []

img = cv2.imread('D:/Downloads/all-item/4.png', 0)

# 水平投影

pos = [0] * img.shape[0]

print(img.shape[0], img.shape[1])
                
mode = 1;

img = delWhite(img)
img = cv2.bitwise_not(img)
img = erodeImg(img)
#cv2.imshow('newimg', img)
#cv2.waitKey()

getTextProjection(img, pos, mode)            
dramProjection(pos, mode)

# 获取切割后的水平图    
peekRange = []
getPeekRange(pos, peekRange)


# 垂直投影      
peekRange2 = []
       
for peek in peekRange:
    rowimg = cutRow(img, peek)
    #cv2.imshow('rowimg', rowimg)
    #cv2.waitKey()
    peekRange2 = cutCol(rowimg)
    pinRectangle(peek, peekRange2, img)

    
cv2.imshow('newimg', img)
cv2.waitKey()
print('end')
            

for p in peekRange:
    pass
    #colimg = rowimg[0:rowimg.shape[1], p[0]:p[1]]
    #cv2.imwrite('D:/Downloads/word/'+ str(index) +'.jpg', colimg)
    #index += 1
    #cv2.imshow('colimg', colimg)
    #cv2.waitKey()         