'''
Created on 2018年4月29日

@author: Administrator
'''
# -*- coding:utf-8 -*- 

from collections import Iterable

def triangles():
    row = [1]
    while True:
        yield row
        row = [1] + [row[i]+row[i+1] for i in range(len(row) - 1)] + [1]
            
for i, x in enumerate(triangles()):
    if i is 10:
        break
    print(x)

t = triangles()
for i in range(10):
    print(next(t))
    
g = (i for i in range(10))

for i in g:
    print(i)
    
print(isinstance(g, Iterable))

# for循环本质上是一个迭代器
itlist = iter(list(range(10)))

while True:
    try:
        print(next(itlist))
    except StopIteration as e:
        print(e.value)
        break;

if __name__ == '__main__':
    print('main')
