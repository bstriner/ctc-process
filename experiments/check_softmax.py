import numpy as np
import tensorflow as tf

x = np.array([
    [
        [1, 1, 1, 3],
        [1, 2, 1, 1],
        [1, 1, 1, 3],
        [3, 2, 3, 1]
    ],
    [
        [1, 6, 7, 3],
        [1, 2, 1, 1],
        [1, 1, 1, 3],
        [3, 2, 3, 1]
    ]
])
x = tf.constant(x, dtype=tf.float32)

lens = tf.constant([3, 3, 3, 2], dtype=tf.int64)

mask = tf.constant([
    [
        [True, True, True, False],
        [True, True, True, False],
        [True, False, True, False],
        [False, False, False, False],
    ],
    [
        [True, True, True, False],
        [True, True, True, False],
        [True, False, True, False],
        [True, True, True, True],
    ]
])

idx = tf.where(mask)
vals = tf.gather_nd(params=x, indices=idx)
sparse = tf.SparseTensor(
    indices=idx,
    values=vals,
    dense_shape=tf.shape(x, out_type=tf.int64)
)

sm = tf.sparse_softmax(sparse)
#dense = tf.sparse_tensor_to_dense(sm, default_value=0)
dense = tf.scatter_nd(
    indices=sm.indices,
    updates=sm.values,
    shape=sm.dense_shape
)

grad = tf.gradients(dense[0,1,2], [vals])[0]

with tf.train.MonitoredSession() as sess:
    print(sess.run(dense))
    print(np.sum(sess.run(dense), axis=-1))
    print(sess.run(grad))
