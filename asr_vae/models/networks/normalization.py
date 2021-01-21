import tensorflow as tf

from .lstm_utils import sequence_index
from .variational.variational_variable import VariationalParams, get_variable


def batch_norm_none(inputs):
    return inputs


def batch_norm_constlen(inputs, norm_scaling=None, is_training=True, renorm=False):
    outputs = tf.contrib.layers.batch_norm(
        inputs=inputs,
        scope='batch_norm',
        is_training=is_training,
        scale=False,
        center=False,
        renorm=renorm
    )
    if norm_scaling is not None:
        outputs = norm_scaling(outputs)
    return outputs


def make_batch_norm_constlen(
        scope='sequence_batch_norm', is_training=True,
        norm_scaling=None, renorm=False
):
    def batch_norm(inputs):
        with tf.variable_scope(scope):
            return batch_norm_constlen(
                inputs=inputs,
                norm_scaling=norm_scaling,
                is_training=is_training,
                renorm=renorm)

    return batch_norm


def make_sequence_batch_norm(
        sequence_lengths, maxlen,
        scope='sequence_batch_norm', is_training=True,
        norm_scaling=None, renorm=False
):
    # norm axis 0,1
    idx = sequence_index(
        lengths=sequence_lengths,
        maxlen=maxlen
    )

    def sequence_batch_norm(inputs):
        with tf.variable_scope(scope):
            if inputs.shape.ndims == 3:
                l = tf.shape(inputs)[0]
                n = tf.shape(inputs)[1]
                d = inputs.shape[2].value
                values = tf.gather_nd(
                    params=inputs,
                    indices=idx
                )
                normed = tf.contrib.layers.batch_norm(
                    inputs=values,
                    scope='batch_norm',
                    is_training=is_training,
                    scale=False,
                    center=False,
                    renorm=renorm
                )
                if norm_scaling is not None:
                    normed = norm_scaling(normed)
                outputs = tf.scatter_nd(
                    updates=normed,
                    indices=idx,
                    shape=[l, n, d]
                )
                return outputs
            else:
                assert inputs.shape.ndims == 2
                return batch_norm_constlen(
                    inputs=inputs,
                    norm_scaling=norm_scaling,
                    is_training=is_training,
                    renorm=renorm
                )

    return sequence_batch_norm


def make_sequence_channel_norm(
        norm_scaling=None, eps=1e-3
):
    # norm axis 2
    def sequence_channel_norm(inputs):
        with tf.variable_scope('channel_norm'):
            mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
            meaned = inputs - mean
            std = tf.reduce_mean(tf.square(meaned), axis=-1, keepdims=True)
            outputs = meaned / tf.sqrt(eps + std)
            if norm_scaling is not None:
                outputs = norm_scaling(outputs)
            return outputs

    return sequence_channel_norm


def make_sequence_layer_norm(
        sequence_lengths, maxlen, norm_scaling=None, eps=1e-3
):
    # norm axis 0,2
    mask = tf.sequence_mask(lengths=sequence_lengths, maxlen=maxlen)
    mask = tf.transpose(mask, (1, 0))
    mask = tf.expand_dims(mask, axis=-1)
    mask = tf.cast(mask, tf.float32)

    def sequence_layer_norm(inputs):
        with tf.variable_scope('sequence_norm'):
            dim = inputs.shape[-1].value
            assert dim
            count = tf.reduce_sum(mask, axis=(0, 2), keepdims=True) * dim
            mean = tf.reduce_sum(inputs * mask, axis=(0, 2), keepdims=True) / count
            meaned = inputs - mean
            std = tf.reduce_sum(tf.square(meaned * mask), axis=(0, 2), keepdims=True)
            outputs = meaned / tf.sqrt(eps + std)
            if norm_scaling is not None:
                outputs = norm_scaling(outputs)
            return outputs

    return sequence_layer_norm


def make_sequence_time_norm(
        sequence_lengths, maxlen, norm_scaling=None, eps=1e-3
):
    # norm axis 0
    mask = tf.sequence_mask(lengths=sequence_lengths, maxlen=maxlen)
    mask = tf.transpose(mask, (1, 0))
    mask = tf.expand_dims(mask, axis=-1)
    mask = tf.cast(mask, tf.float32)

    def sequence_layer_norm(inputs):
        with tf.variable_scope('sequence_norm'):
            count = tf.reduce_sum(mask, axis=0, keepdims=True)
            mean = tf.reduce_sum(inputs * mask, axis=0, keepdims=True) / count
            meaned = inputs - mean
            std = tf.reduce_sum(tf.square(meaned * mask), axis=0, keepdims=True)
            outputs = meaned / tf.sqrt(eps + std)
            if norm_scaling is not None:
                outputs = norm_scaling(outputs)
            return outputs

    return sequence_layer_norm


def make_norm_scaling(
        vparams: VariationalParams,
        is_training=True,
        scale=True
):
    if scale:
        def norm_scaling(inputs):
            with tf.variable_scope('norm_scaling'):
                dim = inputs.shape[-1].value
                beta = get_variable(
                    name='beta',
                    shape=[dim],
                    initializer=tf.initializers.zeros,
                    is_training=is_training,
                    vparams=vparams
                )
                gamma = get_variable(
                    name='gamma',
                    shape=[dim],
                    initializer=tf.initializers.zeros,
                    is_training=is_training,
                    vparams=vparams
                )
                scale = tf.exp(gamma / 2.0)
                return (inputs * scale) + beta
    else:
        def norm_scaling(inputs):
            return inputs
    return norm_scaling


def make_double_norm(is_training=True):
    def norm(inputs):
        if inputs.shape.ndims == 2:
            return batch_norm_constlen(inputs=inputs, is_training=is_training)
        elif inputs.shape.ndims==3:
            return sequence_double_norm(inputs=inputs, is_training=is_training)
        else:
            raise ValueError()

    return norm


def sequence_double_norm(inputs, norm_scaling=None, is_training=True, epsilon=1e-3):
    """

    :param inputs: (l,n,d)
    :param is_training:
    :param epsilon:
    :return:
    """
    assert inputs.shape.ndims == 3
    seq_mean = tf.reduce_mean(inputs, axis=0, keep_dims=True)
    seq_meaned = inputs - seq_mean
    seq_var = tf.reduce_mean(tf.square(seq_meaned), axis=0, keep_dims=True)
    seq_std = tf.sqrt(epsilon + seq_var)
    seq_normed = seq_meaned / seq_std

    batch_feats = seq_mean #tf.concat([seq_mean, seq_std], axis=-1)
    batch_feats_normed = tf.contrib.layers.batch_norm(
        inputs=batch_feats,
        scope='batch_feats_batch_norm',
        is_training=is_training,
        scale=False,
        center=False,
        renorm=False
    )
    l = tf.shape(inputs)[0]
    batch_feats_normed_tiled = tf.tile(tf.expand_dims(batch_feats_normed, 0), [l, 1, 1])
    all_feats = tf.concat([seq_normed, batch_feats_normed_tiled], axis=-1)
    if norm_scaling:
        all_feats = norm_scaling(all_feats)
    return all_feats


def make_sequence_batch_normalization_fn(
        params, sequence_lengths, maxlen, is_training=True
):
    vparams = VariationalParams(
        mode=params.variational_mode,
        sigma_init=params.variational_sigma_init,
        mu_prior=0.0,
        sigma_prior=params.variational_sigma_prior,
        scale=params.variational_scale
    )

    if params.batch_norm == 'none':
        return batch_norm_none
    else:
        norm_scaling = make_norm_scaling(
            is_training=is_training,
            scale=params.batch_norm_scale,
            vparams=vparams
        )
        if params.batch_norm == 'double_norm':
            return
        elif params.batch_norm == 'batch_norm_constlen':
            return make_batch_norm_constlen(
                norm_scaling=norm_scaling,
                is_training=is_training,
                renorm=False
            )
        elif params.batch_norm == 'batch_renorm_constlen':
            return make_batch_norm_constlen(
                norm_scaling=norm_scaling,
                is_training=is_training,
                renorm=True
            )
        elif params.batch_norm == 'batch_norm':
            return make_sequence_batch_norm(
                sequence_lengths=sequence_lengths,
                maxlen=maxlen,
                norm_scaling=norm_scaling,
                is_training=is_training,
                renorm=False
            )
        elif params.batch_norm == 'batch_renorm':
            return make_sequence_batch_norm(
                sequence_lengths=sequence_lengths,
                maxlen=maxlen,
                norm_scaling=norm_scaling,
                is_training=is_training,
                renorm=True
            )
        elif params.batch_norm == 'layer_norm':
            return make_sequence_layer_norm(
                sequence_lengths=sequence_lengths,
                maxlen=maxlen,
                norm_scaling=norm_scaling
            )
        elif params.batch_norm == 'time_norm':
            return make_sequence_time_norm(
                sequence_lengths=sequence_lengths,
                maxlen=maxlen,
                norm_scaling=norm_scaling)
        elif params.batch_norm == 'channel_norm':
            return make_sequence_channel_norm(
                norm_scaling=norm_scaling)
        else:
            raise ValueError("batch_norm")
