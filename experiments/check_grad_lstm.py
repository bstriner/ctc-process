import tensorflow as tf
from tensorflow.contrib.cudnn_rnn.python.layers.cudnn_rnn import CudnnLSTM

n = 32
l = 1000
d=64
x = tf.random_normal(shape=(l, n, d))
layer = CudnnLSTM(num_layers=1, num_units=100)
y, _ = layer(x)
print("Y: {}".format(y))

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    g1 = tf.gradients(tf.reduce_sum(y), [x])[0]
    print("G1: {}".format(g1))
    print("G1: {}".format(sess.run(g1)))

    g2 = tf.sqrt(tf.reduce_sum(tf.square(g1), axis=(0,2)))
    g3 = tf.reduce_mean(tf.square(g2-1))
    g4 = tf.gradients(g3, [layer.kernel])[0]
    print("G4: {}".format(g4))
    print("G4: {}".format(sess.run(g4)))


