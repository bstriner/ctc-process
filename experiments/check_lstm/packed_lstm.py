import numpy as np
import tensorflow as tf
from tensorflow.contrib.cudnn_rnn.python.ops.cudnn_rnn_ops import sequence_gather_scatter_indices_presorted
from tensorflow.contrib.cudnn_rnn.python.layers.cudnn_rnn import CUDNN_RNN_BIDIRECTION, CudnnLSTM
from tensorflow.contrib.framework.python.ops import argsort


def make_initial_states(batch_size, num_layers, dim, bidirectional=False):
    num_dirs = 2 if bidirectional else 1
    initial_c = tf.get_variable(
        name='initial_c',
        dtype=tf.float32,
        initializer=tf.initializers.zeros,
        shape=(num_layers * num_dirs, 1, dim))
    initial_h = tf.get_variable(
        name='initial_h',
        dtype=tf.float32,
        initializer=tf.initializers.zeros,
        shape=(num_layers * num_dirs, 1, dim))
    initial_h, initial_c = [
        tf.tile(s, [1, batch_size, 1])
        for s in [initial_h, initial_c]
    ]
    return initial_h, initial_c

# Load data
info = np.load('data.npz')
num_layers = np.asscalar(info['num_layers'])
num_units = np.asscalar(info['num_units'])
data = info['data']
kernel = info['kernel']
sequence_lengths = info['sequence_lengths']

print("Units: {}".format(num_units))

# Run LSTM
inputs = tf.constant(data, dtype=tf.float32)
batch_order = argsort(sequence_lengths, direction='DESCENDING')
sorted_sequence_lengths = tf.gather(params=sequence_lengths, indices=batch_order)
idx = sequence_gather_scatter_indices_presorted(
    sorted_sequence_lengths=sorted_sequence_lengths,
    batch_order=batch_order)
packed = tf.gather_nd(params=inputs, indices=idx)

lstm = CudnnLSTM(
    num_layers=num_layers,
    num_units=num_units,
    direction=CUDNN_RNN_BIDIRECTION,
)
print("Packed: {}".format(packed))

batch_size = data.shape[1]
initial_states = make_initial_states(
    batch_size=batch_size,
    num_layers=num_layers,
    dim=num_units,
    bidirectional=True)
ypacked, _ = lstm(packed,
               initial_state=initial_states,
               sequence_lengths=sorted_sequence_lengths,
               training=True)
l = data.shape[0]
y = tf.scatter_nd(
    updates=ypacked,
    shape=tf.constant([l,batch_size,num_units*2], dtype=tf.int32),
    indices=idx)
dxdy, dxdw = tf.gradients(tf.reduce_sum(y), [inputs, lstm.kernel])
assignment = tf.assign(lstm.kernel, tf.constant(kernel, dtype=tf.float32))

with tf.train.MonitoredSession() as sess:
    sess.run(assignment)
    _y, _dxdy, _dxdw = sess.run([y, dxdy, dxdw])

np.savez('packed_results.npz',
         y=_y,
         dxdy=_dxdy,
         dxdw=_dxdw)

print(data)
print(_dxdy)