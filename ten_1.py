# -*-coding:utf8-*-
import  tensorflow as tf


#================计算图=====================
x = tf.constant([[2.,3.]])
w = tf.constant([[2.],[2.]])
y = tf.matmul(x,w)
print y

#Tensor("MatMul:0", shape=(1, 1), dtype=float32)
#==========================================


#=============用会话执行计算图中的节点运算====================

'''sess = tf.Session()
print sess.run(y)
sess.close()'''
with tf.Session() as sess:
    print sess.run(y)
#============================================================

w1 = tf.Variable(tf.random_normal)
