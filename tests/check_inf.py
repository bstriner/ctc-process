import numpy as np
import tensorflow as tf


def masked_softmax1(logits, mask):
    _p = tf.nn.softmax(logits, axis=1)
    m = tf.cast(mask, tf.float32)
    m = tf.expand_dims(m, 2)
    _p = _p * m
    _p = _p / tf.reduce_sum(_p, axis=1, keepdims=True)
    return _p


def masked_softmax(logits, mask):
    """
    Masked softmax over dim 1, mask broadcasts over dim 2
    :param logits: (N, L, T)
    :param mask: (N, L)
    :return: probabilities (N, L, T)
    """
    t = tf.shape(logits)[2]
    indices = tf.cast(tf.where(tf.logical_not(mask)), tf.int32)
    inf = tf.constant(np.array([[np.inf]], dtype=np.float32), dtype=tf.float32)
    infs = tf.tile(inf, [tf.shape(indices)[0], t])
    infmask = tf.scatter_nd(
        indices=indices,
        updates=infs,
        shape=tf.shape(logits))
    _p = tf.nn.softmax(logits - infmask, axis=1)
    return _p


# with tf.device("/cpu:0"):
def check_inf(logits, lengths):
    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
        _logits = tf.placeholder(dtype=tf.float32, shape=(None, None, None), name='logits')
        _lengths = tf.placeholder(dtype=tf.int32, shape=(None,), name='lengths')
        mask = tf.sequence_mask(_lengths, maxlen=tf.shape(_logits)[1])
        _p = masked_softmax(_logits, mask)
        _g = tf.gradients(tf.reduce_sum(tf.square(_p)), _logits)
        p, m, g = sess.run([_p, mask, _g], feed_dict={_logits: logits, _lengths: lengths})
        print(p)
        print(m)
        print(g)


def main():
    logits = np.array([[
        [1, 2, 3, 4, 5],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]]]).astype(np.float32)
    lengths = np.array([3]).astype(np.int32)
    p = check_inf(logits=logits, lengths=lengths)

    pass


if __name__ == '__main__':
    main()
