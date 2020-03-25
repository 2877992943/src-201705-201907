# coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding("utf8")

import tensorflow as tf




#nb_iter = tf.constant(value=10)
#This solution does not work at all
nb_iter = tf.get_variable('nb_iter', shape=(), dtype=tf.int32, trainable=False,
                          initializer=tf.zeros_initializer())
nb_iter = tf.add(nb_iter,10)
i = tf.get_variable('i', shape=(), trainable=False,
                     initializer=tf.zeros_initializer(), dtype=nb_iter.dtype)
v = tf.get_variable('v', shape=(10), trainable=False,
                     initializer=tf.random_uniform_initializer, dtype=tf.float32)

loop_condition = lambda i: tf.less(i, nb_iter)
def loop_body(i):
    #i = tf.Print(i, [v[i]], message='Another vector element: ')
    return [tf.add(i, 1)]

i = tf.while_loop(loop_condition, loop_body, [i])

initializer_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(initializer_op)
    res = sess.run(i)
    print('res is now {}'.format(res))

