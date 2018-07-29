'''
Created on 2018年5月16日

@author: Administrator
'''

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

import tensorflow as tf  
import os
import cv2

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

init_op = tf.initialize_all_variables()
init = tf.global_variables_initializer()

rootdir = 'TestImg/'

with tf.Session() as sess:
    sess.run(init)  
    saver = tf.train.Saver()
    saver.restore(sess, "Model/model.ckpt")
    
    for d in os.listdir(rootdir):
        if d.endswith('.png') or d.endswith('.jpg'):
            img = cv2.imread(rootdir + d, 0)
            img = img.reshape(img.shape[0] * img.shape[1])
            prediction = tf.argmax(y, 1)
            predint = prediction.eval(feed_dict={x: [img]}, session=sess)
            print (d, predint)
