'''
Created on 2018年5月15日

@author: Administrator
'''

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

import tensorflow as tf

# 定义graph
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# 定义模型
y = tf.nn.softmax(tf.matmul(x, W) + b)
# 利用交叉熵做损失函数，tf.reduce_sum把minibatch里的每张图片的交叉熵值都加起来了，取均值
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
# 学习步长为0.5，使用梯度下降算法GradientDescentOptimizer最下化损失函数
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# 初始化Variables
init = tf.global_variables_initializer()

# 初始化变量，输入数据，并计算损失函数与利用优化算法更新参数
#with tf.Session() as sess:
sess = tf.Session()
sess.run(init)
for i in range(2000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        
saver = tf.train.Saver()
saver.save(sess, 'Model/model.ckpt')
#  tf.argmax(y,1)返回的是模型对于任一输入x预测到的标签值，
# 而 tf.argmax(y_,1) 代表正确的标签，我们可以用 tf.equal
# 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 将布尔值转换为浮点数来代表对、错，然后取平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

sess.close()

