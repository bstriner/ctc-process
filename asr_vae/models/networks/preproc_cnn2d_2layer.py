import tensorflow as tf


def preproc_cnn2d_2layer(inputs, params, strides=None, is_training=True):
    """

    :param inputs: (n,l, d)
    :param params:
    :param is_training:
    :return:
    """
    if strides is None:
        strides = [(2,2),(1,2)]
    assert len(strides) == 2
    # 32, 41x11, 2x2
    # 32, 21x11, 2x1
    if params.batch_norm == 'none':
        bn = False
    elif params.batch_norm == 'batch_norm_constlen':
        bn = True
    else:
        raise ValueError()
    h = inputs
    h = tf.expand_dims(h, -1)
    assert h.shape.ndims == 4
    # activation = clipped_relu
    activation = tf.nn.leaky_relu
    with tf.variable_scope('preproc'):
        h = tf.layers.conv2d(
            inputs=h,
            filters=32,
            kernel_size=(11, 41),
            strides=strides[0],
            padding='same',
            data_format='channels_last',
            # dilation_rate=(1, 1),
            activation=activation,
            name="preproc_layer_1"
        )
        if bn:
            with tf.variable_scope('bn_1'):
                h = tf.layers.batch_normalization(
                    inputs=h,
                    training=is_training
                )
        h = tf.layers.conv2d(
            inputs=h,
            filters=64,
            kernel_size=(11, 21),
            strides=strides[1],
            padding='same',
            data_format='channels_last',
            # dilation_rate=(1, 1),
            activation=activation,
            name="preproc_layer_2"
        )
        if bn:
            with tf.variable_scope('bn_2'):
                h = tf.layers.batch_normalization(
                    inputs=h,
                    training=is_training
                )
        n = tf.shape(h)[0]
        logit_lengths = tf.shape(h)[1]
        h = tf.reshape(h, [n, logit_lengths, h.shape[2].value * h.shape[3].value])

    logit_lengths = tf.expand_dims(logit_lengths, 0)
    logit_lengths = tf.tile(logit_lengths, [n])

    return h, logit_lengths
