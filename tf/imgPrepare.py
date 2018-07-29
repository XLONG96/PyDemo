'''
Created on 2018年5月16日

@author: Administrator
'''

import cv2
import numpy as np

img = cv2.imread('./TestImg/6_0.png', 0)
a = np.asarray(img)
b = a.reshape(1, 784)

#print(b[0])

nb = np.zeros(784, dtype = int)

for i in range(784):
    if b[0][i] > 0:
        nb[i] = 1

print(nb)

with open('./TestImg/6_0.txt', 'w') as f:
    f.write(str(nb))

#cv2.imshow('img', img)
#cv2.waitKey(0)