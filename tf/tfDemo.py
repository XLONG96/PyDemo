'''
Created on 2018年5月15日

@author: Administrator
'''
import tensorflow as tf

hello = tf.constant('hello tensorflow')
sess = tf.Session()
print(sess.run(hello))
sess.close()


matrix1 = tf.constant([[1, 1, 1], [2, 2, 2]])
matrix2 = tf.constant([[2, 2], [1, 1], [1, 1]])

product = tf.matmul(matrix1, matrix2)


with tf.Session() as sess:
    print(sess.run(product))
    
index = tf.argmax([0,0,1,0,0], 1)
print(index)