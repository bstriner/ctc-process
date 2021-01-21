import tensorflow as tf


def preproc_cnn1d_1layer(inputs, params, is_training=True):
    """

    :param inputs: (n,l, d)
    :param params:
    :param is_training:
    :return:
    """
    # 1280, 11, 2
    h = inputs
    assert h.shape.ndims == 3
    # activation = clipped_relu
    activation = tf.nn.leaky_relu
    with tf.variable_scope('preproc'):
        h = tf.layers.conv1d(
            inputs=h,
            filters=1280,
            kernel_size=(11,),
            strides=(2,),
            padding='same',
            data_format='channels_last',
            # dilation_rate=(1, 1),
            activation=activation,
            name="preproc_layer_1"
        )
        n = tf.shape(h)[0]
        logit_lengths = tf.shape(h)[1]
        logit_lengths = tf.expand_dims(logit_lengths, 0)
        logit_lengths = tf.tile(logit_lengths, [n])

    return h, logit_lengths
