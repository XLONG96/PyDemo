'''
Created on 2018年4月7日

@author: Administrator
'''

def reOrderArray(array):
    odd, even = [],[]
    for i in array:
        if i%2 == 1: 
            odd.append(i) 
        else: even.append(i)
    return odd + even

array = [123,4,4,5,6,23,5,6]

narray = reOrderArray(array)

for i in narray:
    print(i, end=" ")