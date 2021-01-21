import timeit

import numpy as np
import tensorflow as tf

from asr_vae.models.networks.attention import attention_fn_softmax, attention_fn_softmax_constlen, \
    attention_fn_softmax_loop, attention_fn_softmax_loop2

ul = 350
tl = 300
n = 32
dim_in = 128
dim = 64
number = 100
length_variance = 50
uh = tf.random_normal(shape=(ul, n, dim_in))
th = tf.random_normal(shape=(tl, n, dim_in))

uls = tf.random_uniform(minval=ul - length_variance, maxval=ul, shape=(n,), dtype=tf.int32)
tls = tf.random_uniform(minval=tl - length_variance, maxval=tl, shape=(n,), dtype=tf.int32)

um = tf.transpose(tf.sequence_mask(uls), (1, 0))
tm = tf.transpose(tf.sequence_mask(tls), (1, 0))

with tf.variable_scope('a1'):
    a1 = attention_fn_softmax(
        utterance_h=uh,
        utterance_mask=um,
        transcript_h=th,
        transcript_mask=tm,
        dim=dim
    )
    g1 = tf.gradients(a1[0,0,0], [uh])[0]

with tf.variable_scope('a1', reuse=True):
    a2 = attention_fn_softmax_loop(
        utterance_h=uh,
        utterance_lengths=uls,
        transcript_h=th,
        transcript_lengths=tls,
        dim=dim
    )
with tf.variable_scope('a1', reuse=True):
    a3 = attention_fn_softmax_loop2(
        utterance_h=uh,
        utterance_lengths=uls,
        transcript_h=th,
        transcript_lengths=tls,
        dim=dim
    )
    g3 = tf.gradients(a3[0,0,0],[uh])[0]

with tf.variable_scope('a1', reuse=True):
    a4 = attention_fn_softmax_constlen(
        utterance_h=uh,
        transcript_h=th,
        dim=dim
    )

with tf.train.MonitoredSession() as sess:
    x1, x2, x3, y1, y3 = sess.run([a1, a2, a3, g1, g3])
    print(x1.shape)
    print(x2.shape)
    print(x1[0, :5, :5])
    print(x2[0, :5, :5])
    print(np.argmax(x1[0, 0, :]))
    print(np.argmax(x2[0, 0, :]))
    print(np.max(x1[0, 0, :]))
    print(np.max(x2[0, 0, :]))
    assert np.allclose(x1, x2, atol=1e-5)
    assert np.allclose(x1, x3, atol=1e-5)
    assert np.allclose(y1, y3, atol=1e-5)


    def s1():
        sess.run([a1])


    def s2():
        sess.run([a2])


    def s3():
        sess.run([a3])


    def s4():
        sess.run([a4])


    n1 = timeit.timeit(stmt=s1, number=number) / number
    n2 = timeit.timeit(stmt=s2, number=number) / number
    n3 = timeit.timeit(stmt=s3, number=number) / number
    n4 = timeit.timeit(stmt=s4, number=number) / number
    print("N1: {}".format(n1))
    print("N2: {}".format(n2))
    print("N3: {}".format(n3))
    print("N4: {}".format(n4))
