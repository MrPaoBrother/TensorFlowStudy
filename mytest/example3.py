# -*- coding:utf8 -*-

import tensorflow as tf

from numpy.random import RandomState

x = tf.placeholder(tf.float32 , shape = (None , 2) , name = 'x-input')

y_ = tf.placeholder(tf.float32 , shape = (None , 1) , name = 'y-input')
all_datas = 128 #所有的数组
W1 = tf.Variable(tf.random_normal([2,3] , stddev = 1 , seed = 1))
W2 = tf.Variable(tf.random_normal([3,1] , stddev = 1 , seed = 1))

a = tf.matmul(x , W1)

y = tf.matmul(a , W2)

rdm = RandomState(1)

X = rdm.rand(all_datas , 2)

Y = [[int(x0 + x1 < 1)] for (x0 , x1) in X]

#=================下面是反向传播==================

cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y , 1e-10 , 1)))
learning_rate = 0.001
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

STEPS = 5000
batch_size = 8 #一批八个
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    for i in range(STEPS):
	start = (i * batch_size) % all_datas
	end = min(start + batch_size , all_datas)
	sess.run(train_step , feed_dict = {x:X[start:end] , y_:Y[start:end]})
	if i % 1000 == 0:
	    current_cross_entropy = sess.run(cross_entropy , feed_dict = {x:X , y_:Y})
	    print "current nums are:%d , and value = %g\n"%(i , current_cross_entropy)
    print "==============W1================="
    print sess.run(W1)
    print "==============W2================="
    print sess.run(W2)
print "end"
	
