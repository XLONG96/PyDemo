'''
Created on 2018年5月4日

@author: Administrator
'''

import cv2
import numpy

def CLAMP_0_255(pixel):
    if pixel > 255:
        pixel = 255
    if pixel < 0:
        pixel = 0
    return pixel

def USMSharp(img, nAmount = 200):
    sigma = 3
    threshold = 1
    amount = nAmount / 100
    
    blur = cv2.GaussianBlur(img, (31, 31), sigma)
    
    lowContrastMask = abs(img - blur) < threshold
    dst = img * (1 + amount) + blur * (-amount)
    
    return dst
    

# 参考白
def white(img):
    thresholdco = 0.05
    #thresholdnum = 100
    histogram = [0] * 256
    for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                b = img[i, j, 0]
                g = img[i, j, 1]
                r = img[i, j, 2]
                # 计算灰度值
                gray = int((r * 299 + g * 587 + b * 114) / 1000)
                #print(gray)
                histogram[gray] += 1
    
    print(histogram[::])
    calnum = 0
    num = 0
    total = img.shape[0] * img.shape[1]
    # 得到满足系数thresholdco的临界灰度级
    for i in range(256):
        if calnum / total < thresholdco:
            print(calnum / total, i)
            calnum += histogram[255 - i]
            num = i
        else:
            break
        
    averagegray = 0
    calnum = 0
    
    #for i in range(255 - num, 256):
    #    averagegray += histogram[i] * i
    #    calnum += histogram[i]
    
    i = 255
    while i >= 255 - num:
        averagegray += histogram[i] * i
        calnum += histogram[i]
        i -= 1
    
    print(averagegray, calnum)
    averagegray /= calnum
    
    # 光线补偿系数
    co = 255.0 / averagegray
    print(averagegray, calnum, co)
    
    for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[i, j, 0] = CLAMP_0_255(co * img[i, j, 0] + 0.5)
                img[i, j, 1] = CLAMP_0_255(co * img[i, j, 1] + 0.5)
                img[i, j, 2] = CLAMP_0_255(co * img[i, j, 2] + 0.5)
    return img


# USM锐化
def usm(img, tdel):
    # 消除透明背景
    if len(img.shape) >= 3 and img.shape[2] == 4:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if not img[i, j, 3]:
                    img[i, j, 0] = 255
                    img[i, j, 1] = 255
                    img[i, j, 2] = 255
    
    
    # 参考白
    #white(img)
    # 获取灰度图
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #USMSharp(img)
    
    height = img.shape[0]
    width = img.shape[1]
    totalGray, blackNum = 0, 0
    for i in range(height):
        for j in range(width):
            if (j > 0 and img[i, j] + tdel < img[i, j - 1]) \
            and (j + 1) < height and img[i, j] + tdel < img[i, j + 1]:
                totalGray += img[i, j]
                blackNum += 1
    
    for i in range(width):
        for j in range(height):
            if j > 0 and img[j, i] + tdel < img[j - 1, i] and \
                j + 1 < height and img[j, i] + tdel < img[j + 1, i]:
                    totalGray += img[j, i]
                    blackNum += 1
    
    th = totalGray / blackNum
    # 二值化
    ret, th1 = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)            
    
    return th1


if __name__ == '__main__': 
    root = 'D:/Downloads/all-item/'
    for i in range(1, 51):
        try:
            print(root + str(i) + '.png')
            img = cv2.imread(root + str(i) + '.png', -1)
            img = usm(img, 7)
            #cv2.imshow('img', usm)
            #cv2.waitKey()
            print(str(i) + ' finished!')
            cv2.imwrite(root + 'fix-item/%06d.png' % i, img)
        except Exception as e:
            print(e)
            print('File open exception!')
            continue


