import numpy as np
import tensorflow as tf
from tensorflow.contrib.cudnn_rnn.python.layers.cudnn_rnn import CUDNN_RNN_BIDIRECTION, CudnnLSTM

# Setup
input_dim = 4
num_layers = 2
num_units = 3
sequence_lengths = np.array([5, 3, 2], dtype=np.int32)

# Generate data
maxlen = np.max(sequence_lengths)
n = sequence_lengths.shape[0]
data = np.stack([
    np.pad(np.random.normal(loc=0, scale=1, size=(l, 4)), [(0, maxlen - l), (0, 0)], mode='constant')
    for l in sequence_lengths], axis=1)

lstm = CudnnLSTM(
    num_layers=num_layers,
    num_units=num_units,
    direction=CUDNN_RNN_BIDIRECTION,
)
x = tf.placeholder(dtype=tf.float32, shape=(None, None, input_dim))
y, rets = lstm(x)
k = lstm.kernel

with tf.train.MonitoredSession() as sess:
    kernel = np.random.normal(0, 1, size=sess.run(k).shape)

info = {
    "data": data,
    "kernel": kernel,
    "sequence_lengths": sequence_lengths,
    "num_units": num_units,
    "num_layers": num_layers
}
np.savez("data.npz", **info)
