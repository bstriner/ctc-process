import tensorflow as tf


def postproc_cnn1d(inputs, params, strides, is_training=True):
    """

    :param inputs: (l,n, d)
    :param params:
    :param is_training:
    :return:
    """
    # 1280, 11, 2
    if params.batch_norm == 'none':
        bn = False
    elif params.batch_norm == 'batch_norm_constlen':
        bn = True
    else:
        raise ValueError()
    h = inputs
    assert h.shape.ndims == 3
    h = tf.transpose(h, (1, 0, 2))
    h = tf.expand_dims(h, 2)  # (n,l,1,d)
    activation = tf.nn.leaky_relu
    with tf.variable_scope('preproc'):
        for i, stride in enumerate(strides):
            h = tf.layers.conv2d_transpose(
                inputs=h,
                filters=params.decoder_dim,
                kernel_size=(5, 1),
                strides=(stride, 1),
                padding='same',
                data_format='channels_last',
                activation=activation,
                use_bias=True,
                name='postproc_conv2d_{}'.format(i)
            )
            if bn:
                with tf.variable_scope('bn_{}'.format(i)):
                    h = tf.layers.batch_normalization(
                        inputs=h,
                        training=is_training
                    )
        n = tf.shape(h)[0]
        logit_lengths = tf.shape(h)[1]
        logit_lengths = tf.expand_dims(logit_lengths, 0)
        logit_lengths = tf.tile(logit_lengths, [n])
        h = tf.squeeze(h, 2)  # (n,l,d)
        h = tf.transpose(h, [1, 0, 2])

    return h, logit_lengths
