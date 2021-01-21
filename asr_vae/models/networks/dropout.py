import tensorflow as tf


def uout_3d(inputs, p):
    assert inputs.shape.ndims == 3
    n = tf.shape(inputs)[1]
    d = inputs.shape[2].value
    mask = tf.random.uniform(
        minval=1. - p,
        maxval=1. + p,
        dtype=tf.float32,
        shape=(1, n, d)
    )
    outputs = mask * inputs
    return outputs


def dropout_3d(inputs, p):
    assert inputs.shape.ndims == 3
    n = tf.shape(inputs)[1]
    d = inputs.shape[2].value
    mask = tf.random.uniform(
        minval=0.,
        maxval=1.,
        dtype=tf.float32,
        shape=(1, n, d)
    )
    mask = tf.cast(tf.greater(mask, p), dtype=tf.float32) / (1. - p)
    outputs = mask * inputs
    return outputs


def uout_2d(inputs, p):
    assert inputs.shape.ndims == 2
    n = tf.shape(inputs)[0]
    d = inputs.shape[1].value
    mask = tf.random.uniform(
        minval=1. - p,
        maxval=1. + p,
        dtype=tf.float32,
        shape=(n, d)
    )
    outputs = mask * inputs
    return outputs


def dropout_2d(inputs, p):
    assert inputs.shape.ndims == 2
    n = tf.shape(inputs)[0]
    d = inputs.shape[1].value
    mask = tf.random.uniform(
        minval=0.,
        maxval=1.,
        dtype=tf.float32,
        shape=(n, d)
    )
    mask = tf.cast(tf.greater(mask, p), dtype=tf.float32) / (1. - p)
    outputs = mask * inputs
    return outputs


def dropout_nd(inputs, p):
    if inputs.shape.ndims == 2:
        return dropout_2d(inputs, p)
    elif inputs.shape.ndims == 3:
        return dropout_3d(inputs, p)
    else:
        raise ValueError()


def uout_nd(inputs, p):
    if inputs.shape.ndims == 2:
        return uout_2d(inputs, p)
    elif inputs.shape.ndims == 3:
        return uout_3d(inputs, p)
    else:
        raise ValueError()


def make_dropout_fn(is_training, dropout, uout):
    if is_training and dropout > 0.:
        if uout:
            def dropout_fn(inputs):
                return uout_nd(inputs, dropout)
        else:
            def dropout_fn(inputs):
                return dropout_nd(inputs, dropout)
    else:
        def dropout_fn(inputs):
            return inputs
    return dropout_fn


def make_dropout_params_fn(is_training, params):
    return make_dropout_fn(is_training=is_training, dropout=params.decoder_dropout, uout=params.decoder_uout)
