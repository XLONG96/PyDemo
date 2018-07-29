'''
Created on 2018年4月25日

@author: Administrator
'''

import cv2
import numpy as np
from math import ceil

def balance(img, blockSize):
    rows = img.shape[0]
    cols = img.shape[1]
    average = cv2.mean(img)[0]
    rows_new = ceil(rows / blockSize)
    cols_new = ceil(cols / blockSize)
    
    print(rows)
    print(cols)
    print(average)
    
    blockImage = np.zeros((rows_new, cols_new), dtype=img.dtype)
    #cv2.imshow('np',blockImage)
    
    for i in range(0, rows_new):
        for j in range(0, cols_new):
            rowmin = i * blockSize
            rowmax = (i + 1) * blockSize
            if rowmax > rows: 
                rowmax = rows
            colmin = j * blockSize
            colmax = (j + 1) * blockSize
            if colmax > cols: 
                colmax = cols
            
            #print(rowmin)
            imageROI = np.zeros((rowmax - rowmin + 1, colmax - colmin + 1), dtype=img.dtype)
            x, y =0, 0
            for k in range(rowmin, rowmax):
                y = 0 
                for z in range(colmin, colmax):
                    #print(k, z)
                    imageROI[x, y] = img[k, z]
                    y += 1
                x += 1
            #imageROI = img[range(rowmin, rowmax), range(colmin, colmax)]
            temaver = cv2.mean(imageROI)[0]
            blockImage[i, j] = temaver
            print(temaver)
    
    blockImage = blockImage - average
   
    # 双立方差值法
    blockImage2 = cv2.resize(blockImage, (cols, rows), cv2.INTER_CUBIC)
    
    cv2.imshow('a', blockImage2)
    dst = img - blockImage2
    
    cv2.imshow('b', dst)
    

img = cv2.imread('D:/Downloads/all-item/16.jpg', 0)
#balance(img, 2)
print(img.dtype)
cv2.waitKey(0)
cv2.destroyAllWindows()









