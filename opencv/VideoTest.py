'''
Created on 2018年4月18日

@author: Administrator
'''
import cv2

cap = cv2.VideoCapture('D:/Downloads/517736_4b5eec9080671581c88a4199ac4930f1_2.mp4')

while True:
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('frame', gray)
    
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()