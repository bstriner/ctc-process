import tensorflow as tf
import numpy as np

mu = tf.placeholder(
    name='mu',
    shape=[None, None],
    dtype=tf.float32
)

logsigmasq = tf.placeholder(
    name='logsigmasq',
    shape=[None, None],
    dtype=tf.float32
)

kl = -0.5 * (1 + logsigmasq - tf.square(mu) - tf.exp(logsigmasq))
kl = tf.reduce_sum(kl, axis=-1)
kl = tf.reduce_mean(kl, axis=0)

shape = [1, 128]
with tf.train.MonitoredSession() as sess:
    for i in range(-1000, 100):
        nmu = np.zeros(shape=shape, dtype=np.float32)
        nlogsigmasq = np.ones(shape=shape, dtype=np.float32)*i

        actual_kl =  sess.run(kl, feed_dict={mu:nmu, logsigmasq:nlogsigmasq})
        print("{}: {}".format(i, actual_kl))