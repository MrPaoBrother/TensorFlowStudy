# -*-coding:utf8-*-

import tensorflow as tf

from numpy.random import RandomState

'''rdm = RandomState(1)
print rdm

with tf.Session() as sess:
    print sess.run(rdm)'''
batch_size = 8 #一组八个数据
x = tf.placeholder(tf.float32 , shape=(None , 2) , name='x-input')
y_ = tf.placeholder(tf.float32 , shape=(None , 1) , name='y-input')
W1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
a = tf.matmul(x,W1)
W2 = tf.Variable(tf.random_normal([3,1] , stddev=1,seed=1))
y = tf.matmul(a,W2)

rdm = RandomState(1)
data_size = 128
X = rdm.rand(data_size , 2)
Y = [[int(x0 + x1 < 1)] for (x0,x1) in X]

'''print X
print "=============================="
print Y'''
#上面定义了正向传播过程

cross_entropy = -tf.reduce_mean(
	y_ * tf.log(tf.clip_by_value(y , 1e-10 , 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    print "==========W1============"
    print sess.run(W1)
    print "======W2==============="
    print sess.run(W2)
    STEPS = 5000
    for i in range(STEPS):
	start = (i * batch_size)%data_size
	end = min(start + batch_size , data_size)
	sess.run(train_step , feed_dict={x:X[start:end],y_:Y[start:end]})        
	if i % 1000 == 0:
	    total_cross_entropy = sess.run(cross_entropy , feed_dict = {x:X , y_:Y})
	    print "训练%d轮之后,交叉熵在所有数据上的值是%g"%(i , total_cross_entropy)
    print "=========训练结束后(W1)================"
    print sess.run(W1)
    
    print "=========训练结束后(W2)================"
    print sess.run(W2)
print "程序结束.............."


















    

