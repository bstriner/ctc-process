
import tensorflow as tf
from asr_vae.anneal import calc_scale
step = tf.placeholder(dtype=tf.int32, shape=[], name='step')
scale = calc_scale(
    min_val=-2,
    max_val=0,
    start_step=1000,
    end_step=10000,
    step=step
)
with tf.train.MonitoredSession() as sess:
    print(sess.run(scale, feed_dict={step: 0}))
    print(sess.run(scale, feed_dict={step: 500}))
    print(sess.run(scale, feed_dict={step: 1000}))
    print(sess.run(scale, feed_dict={step: 1200}))
    print(sess.run(scale, feed_dict={step: 5000}))
    print(sess.run(scale, feed_dict={step: 15000}))
    pass
