'''
Created on 2018年6月8日

@author: Administrator
'''
num = 1000000000
num3 = 3
num5 = 5
num15 = 15
sum = 0
sum2 = 0
for i in range(num):
    if num3 < num:
        sum += num3
        num3 += 3
    if num5 < num:
        sum += num5
        num5 += 5
    if num15 < num:
        sum2 += num15
        num15 += 15
    if num3 >= num and num5 >= num and num15 >= num:
        break 

print(sum - sum2)