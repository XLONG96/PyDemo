'''
Created on 2018年6月7日

@author: Administrator
'''
import pickle
import cv2
import numpy as np
import random

rootdir = "D:\\Downloads\\crnn\\"
srcdir = "D:\\Downloads\\voc3\\msyahei2\\train\\"
labeldict = {}
chardict = {}
count = 0

# 注意，chinese_labels里面的映射关系是：（ID：汉字）
def get_label_dict():
    f = open('./chinese_labels_new','rb')
    label_dict = pickle.load(f)
    f.close()
    return label_dict

def get_char_dict():
    with open('./chars/char_std_5990.txt', 'r', encoding='utf-8') as f:
        str = f.read()
    chars = str.split('\n')
    indexs = list(range(len(chars)))
    return dict(zip(chars, indexs))

def set_count():
    global count
    try:
        with open(rootdir + 'count.txt', 'r') as f:
            scount = f.read()
            count = int(scount)
    except:
        count = 0

def write_count():
    global count
    with open(rootdir + 'count.txt', 'w') as f:
        f.write(str(count))
       
def write_train(filename, labels):
    with open(rootdir + 'train.txt', 'a') as f:
        f.write("%06d.jpg" %filename)
        for i in labels:
            f.write(" " + str(i))
        f.write("\n")

def appendImg(img, subimg, x, y):    
    img[x:x + subimg.shape[0], y:y + subimg.shape[1]] = subimg

# 0-03754:word  03755-03764:num  03765-03790:letter 3791-3792:() 3793-3794:symbol
def getSubImg():
    tword = 20
    tnum = tword + 10
    tletter = tnum + 0
    tsymbol = tletter + 0
    dstClass = random.randint(0, tsymbol)
    
    if dstClass <= tword:
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
    
    dstname = random.randint(0, 0)
    fontImg = cv2.imread(srcdir + "%05d\\%d.png" %(dstdir, dstname), 0)
    
    if fontTag == 'num' or fontTag == 'letter':
        fontImg = cv2.resize(fontImg, (fontImg.shape[1]//2, fontImg.shape[0]))
    
    ch = labeldict[dstdir]
    
    print("dstdir-%d dstname-%d fontTag-%s" %(dstdir, dstname, fontTag))
    return [fontImg, ch]
  
def create_img():
    global count
    ## create main image in random
    height = 30
    width = 300
    #img = np.ones([height, width]) * 255
    img = np.zeros([height, width], np.uint8)
    name = count
    count += 1

    ## append chile img
    # word count
    column = 10
    margin = 6
    step = 1
    lastwidth = 0
    labels = []

    while column:
        sub = getSubImg()
        subimg = sub[0]
        ch = sub[1]
        lab = chardict.get(ch)
        print("append-char: %s label: %s" %(ch, lab))
        if lab == None:
            continue
        labels.append(lab)
        
        sheight = height - margin
        swidth = int(subimg.shape[1]*(sheight/subimg.shape[0]))
        subimg = cv2.resize(subimg, ( swidth, sheight ))
        if (margin//2 + lastwidth + swidth) > width:
            break
        appendImg(img, subimg, margin//2, lastwidth + margin//2)
        print(sheight, swidth, margin//2, lastwidth + margin//2)

        lastwidth += swidth + step
        column -= 1

    print("height-%d column-%d" %(height, column))
    write_train(name, labels)
    ret, th1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    img = th1
    img = cv2.resize(img, (280, 32))
    cv2.imshow('img-%d' %name, img)
    cv2.imwrite(rootdir + 'images\\' + '%06d.jpg' %name, img)
    cv2.waitKey()
    

if __name__ == '__main__':
    chardict = get_char_dict()
    
    labeldict = get_label_dict()
    
    #print(chardict, '\n', labeldict)
    
    set_count()
    
    for i in range(10):
        create_img()

    write_count()


