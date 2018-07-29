'''
Created on 2018年6月7日

@author: Administrator
'''
'''
import time
N = 1000
for i in range(N):
    print("进度:{0}%".format(round((i + 1) * 100 / N)), end='\x0D')
    #time.sleep(0.01)


import time
import progressbar
p = progressbar.ProgressBar()
N = 1000
for i in p(range(N)):
    time.sleep(0.01)
'''