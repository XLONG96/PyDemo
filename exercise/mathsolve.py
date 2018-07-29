'''
Created on 2018年6月6日

@author: Administrator
'''

import numpy as np
from numpy.linalg.linalg import inv
from numpy import matrix
import math
from math import cos

a = np.array([[1,1],[2,1]])
b= np.array([[1,2],[1,1]])
a = matrix(a)
b = matrix(b)
print(a*b)

ai = a.I
print(ai)
print(a.T)

ainv = inv(a)

print(ainv)

b = np.array([[1, 2], [3, 1]])
b = matrix(b)
bi = b.I
print(inv(b), bi)

print(math.cos(math.pi/8))

B = matrix([[1,0.9,0.81],[1,2.2,4.8],[1,3,9],[1,4,16],[1,5,25]])
Y = matrix([[1.8,3,2.5,3,2]])
A = (B.T*B).I*B.T*Y.T
print(A)

print(Y.T - B*A)
p = math.pi
G = matrix([[0.5, 0.5, 0.5, 0.5],
            [0.707*cos(p/8), 0.707*cos(3*p/8),0.707*cos(7*p/8),0.707*cos(p*9/8)],
            [0.707*cos(2*p/8),0.707*cos(6*p/8),0.707*cos(14*p/8),0.707*cos(18*p/8)],
            [0.707*cos(3*p/8),0.707*cos(9*p/8),0.707*cos(21*p/8),0.707*cos(27*p/8)]
            ])
f = matrix([[1,0,0,1],
            [1,0,0,1],
            [1,0,0,1],
            [1,1,1,1]
            ])

F = G*f*G.T
print(F)

F = matrix([[2.5,0,1.5,0],
            [-0.66,0,0.66,0],
            [0.5,0,-0.5,0],
            [-0.27,0,0.27,0]
    ])
f = G*F*G.T
print(f)


F = matrix([[2.368,-0.471,1.624,0.323],
            [-0.471,0.094,0.323,-0.064],
            [1.624,0.323,0.449,0.089],
            [0.323,-0.641,-0.089,-0.018]  
    ])

G = matrix([[0.5,0.653,0.5,0.271],
            [0.5,0.271,-0.5,-0.653],
            [0.5,-0.271,-0.5,0.653],
            [0.5,-0.653,0.5,-0.271]
    ])

f = G*F*G.T
print(f)

f = matrix([[1,1,1,1],
            [1,0,0,1],
            [1,0,0,1],
            [1,1,1,1]
    ])

F = G*f*G.T
print(F)

f = G*F*G.T
print(f)