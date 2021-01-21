import tensorflow as tf
from tensorflow.contrib import slim

from .lstm_utils import bilstm


def discriminator_varlen_network(
        utterances_t,
        utterance_lengths,
        latent,
        params,
        is_training,
        weight_regularizer=None
):
    assert latent.shape.ndim == 3

    inputs = tf.concat([utterances_t, latent], axis=-1)

    with tf.variable_scope('discriminator_bilstm'):
        _, h = bilstm(
            x=inputs,
            num_layers=params.discriminator_depth,
            num_units=params.discriminator_dim,
            sequence_lengths=utterance_lengths,
            is_training=is_training,
            kernel_regularizer=weight_regularizer
        )  # (N,D)
        h = slim.fully_connected(
            inputs=h,
            num_outputs=1,
            activation_fn=None,
            scope='mlp_output',
            weights_regularizer=weight_regularizer
        )
        h = tf.squeeze(h, axis=-1)  # (N,)
    return h
