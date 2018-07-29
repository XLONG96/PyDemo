'''
Created on 2018年5月29日

@author: Administrator
'''
import cv2
import numpy as np
import random

rootdir = "D:\\Downloads\\voc3\\"
srcdir = "D:\\Downloads\\voc3\\msyahei2\\train\\"
count = 0

def setCount():
    global count
    try:
        with open(rootdir + 'count.txt', 'r') as f:
            scount = f.read()
            count = int(scount)
    except:
        count = 0

def writeCount():
    global count
    with open(rootdir + 'count.txt', 'w') as f:
        f.write(str(count))

def writeImageSet(name):
    subdir = 'ImageSets\\Main\\'
    with open(rootdir + subdir + 'train.txt', 'a') as f:
        f.write(str("%06d\n" %name))
    
    with open(rootdir + subdir + 'trainval.txt', 'a') as f:
        f.write(str("%06d\n" %name))
    
    with open(rootdir + subdir + 'val.txt', 'a') as f:
        f.write(str("%06d\n" %name))

def resetFile():
    with open(rootdir + 'count.txt', 'w') as f:
        f.write("0")
        
    subdir = 'ImageSets\\Main\\'
    with open(rootdir + subdir + 'train.txt', 'w') as f:
        f.write("")
    
    with open(rootdir + subdir + 'trainval.txt', 'w') as f:
        f.write("")
    
    with open(rootdir + subdir + 'val.txt', 'w') as f:
        f.write("")

def writeHead(name, width, height, depth):
    ## create file
    f = open(rootdir + 'Annotations\\' + "%06d.xml" %name, 'w')    
    f.write('<annotation>\n')
    f.write('\t<folder>voc3</folder>\n')
    f.write('\t<filename>%06d.jpg</filename>\n' %name)
    f.write('\t<path>%s%06d.jpg</path>\n' %(rootdir, name))
    f.write('\t<source>\n')
    f.write('\t\t<database>Unknown</database>\n')
    f.write('\t</source>\n')
    f.write('\t<size>\n')
    f.write('\t\t<width>%s</width>\n' %width)
    f.write('\t\t<height>%s</height>\n' %height)
    f.write('\t\t<depth>%s</depth>\n' %depth)
    f.write('\t</size>\n')
    f.write('\t<segmented>0</segmented>\n')   
    f.close()
    
def writeTail(name):
    with open(rootdir + 'Annotations\\' + "%06d.xml" %name, 'a') as f:
        f.write('</annotation>\n')   
   
def writeObj(name, tag, xmin, ymin, xmax, ymax):
    with open(rootdir + 'Annotations\\' + "%06d.xml" %name, 'a') as f:
        f.write('\t<object>\n')   
        f.write('\t\t<name>%s</name>\n' %tag) 
        f.write('\t\t<pose>Unspecified</pose>\n') 
        f.write('\t\t<truncated>1</truncated>\n') 
        f.write('\t\t<difficult>0</difficult>\n') 
        f.write('\t\t<bndbox>\n') 
        f.write('\t\t\t<xmin>%s</xmin>\n' %xmin) 
        f.write('\t\t\t<ymin>%s</ymin>\n' %ymin) 
        f.write('\t\t\t<xmax>%s</xmax>\n' %xmax) 
        f.write('\t\t\t<ymax>%s</ymax>\n' %ymax) 
        f.write('\t\t</bndbox>\n') 
        f.write('\t</object>\n')  

def appendImg(img, subimg, x, y):
    
    img[x:x + subimg.shape[0], y:y + subimg.shape[1]] = subimg
    
    '''
    for i in range(subimg.shape[0]):
        for j in range(subimg.shape[1]):
            img[i + x, j + y] = subimg[i, j]
    '''
# 0-10:word  11-20:num  21-30:letter 
# 0-03754:word  03755-03764:num  03765-03790:letter 3791-3792:() 3793-3794:symbol
# 轮盘算法  
#    pword = 3754/3816
#    pnum = pword + (3764-3755)/3816
#    pletter = pnum + (3816-3765)/3816
#    tword = pword * 6
#    tnum = pnum * 6
#    tletter = pletter * 6
#    dstClass = random.randint(0, 6)
wordcount = 0     
def getSubImg():
    global wordcount
    tspace = -1
    tword = tspace + 1
    tnum = tword + 0
    tletter = tnum + 0
    tsymbol = tletter + 0
    dstClass = random.randint(0, tsymbol)
    if dstClass <= tspace:
        fontTag = 'space'
    elif dstClass <= tword:
        dstdir = random.randint(0, 3754)
        '''
        dstdir = wordcount
        wordcount += 1
        if wordcount >= 3754:
            wordcount = 3754
        '''
        fontTag = 'word'
    elif dstClass <= tnum:
        dstdir = random.randint(3755, 3764)
        fontTag = 'num'
    elif dstClass <= tletter:
        dstdir = random.randint(3765, 3792)
        fontTag = 'letter'
    else:
        dstdir = random.randint(3793, 3794)
        fontTag = 'symbol'
        
    if fontTag == 'space':
        fontImg = np.zeros([50, 50])
        return [fontImg, fontTag]
    dstname = random.randint(0, 0)
    fontImg = cv2.imread(srcdir + "%05d\\%d.png" %(dstdir, dstname), 0)
    if fontTag == 'num' or fontTag == 'letter':
        fontImg = cv2.resize(fontImg, (fontImg.shape[1]//2, fontImg.shape[0]))
        
    print("dstdir-%d dstname-%d fontTag-%s" %(dstdir, dstname, fontTag))
    return [fontImg, fontTag]
  
def crtImg():
    global count
    ## create main image in random
    height = random.randint(26, 36)
    width = 1024
    #img = np.ones([height, width]) * 255
    img = np.zeros([height, width], np.uint8)
    name = count
    count += 1
    writeHead(name, width, height, 1)

    ## append chile img
    # word count
    column = random.randint(12, 30)
    margin = 6
    step = 0
    lastwidth = step
    for i in range(column):
        sub = getSubImg()
        subimg = sub[0]
        tag = sub[1]
        sheight = height - margin
        swidth = int(subimg.shape[1]*(sheight/subimg.shape[0]))
        subimg = cv2.resize(subimg, ( swidth, sheight ))
        if (margin//2 + lastwidth + swidth) > width:
            break
        appendImg(img, subimg, margin//2, lastwidth + margin//2)
        print(sheight, swidth, margin//2, lastwidth + margin//2)
        if tag == 'space':
            lastwidth += swidth
            continue
        xmin = margin//2 + lastwidth
        ymin = margin//2
        xmax = xmin + swidth
        ymax = ymin + sheight
        print(xmin, ymin, xmax, ymax)
        writeObj(name, tag, xmin, ymin, xmax, ymax)
        lastwidth += swidth + step
    
    writeTail(name)
    writeImageSet(name)
    print("height-%d column-%d" %(height, column))
    #cv2.imshow('sd', img)
    #cv2.waitKey()
    #ret, th1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #img = th1
    img = 255 - img
    #cv2.imshow('img-%d'%name, img)
    cv2.imwrite(rootdir + 'JPEGImages\\' + '%06d.jpg'%name, img)
    #cv2.waitKey()

if __name__ == '__main__':
    reset = False
    if reset:
        resetFile()
    setCount()
    for i in range(100):
        crtImg()
    writeCount()
