import tensorflow as tf

from asr_vae.models.networks.variational.variational_lstm import CUDNN_RNN_BIDIRECTION, variational_lstm
tf.compat.v1.disable_eager_execution()
x = tf.random.normal(shape=(10, 5, 6))
y, (h,c) = variational_lstm(
    inputs=x,
    num_units=8,
    scope='mylstm',
    num_layers=3,
    direction=CUDNN_RNN_BIDIRECTION
)
print("y: {}".format(y))
print("h: {}".format(h))
print("c: {}".format(c))

with tf.train.MonitoredSession() as sess:
    outputs, = sess.run([y])
    print(outputs.shape)
