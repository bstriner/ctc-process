import tensorflow as tf
from tensorflow.contrib import slim

from .lstm_utils import bilstm


def discriminator_constlen_network(
        utterances_t,
        utterance_lengths,
        latent,
        params,
        is_training,
        weight_regularizer=None
):
    assert latent.shape.ndims == 2
    with tf.variable_scope('utterances_bilstm'):
        _, utterance_hidden = bilstm(
            x=utterances_t,
            num_layers=params.discriminator_depth,
            num_units=params.discriminator_dim,
            sequence_lengths=utterance_lengths,
            is_training=is_training,
            kernel_regularizer=weight_regularizer
        )  # (N,D)

    with tf.variable_scope('discriminator_outputs'):
        h = tf.concat([utterance_hidden, latent], axis=-1)
        for i in range(params.encoder_depth):
            h = slim.fully_connected(
                inputs=h,
                num_outputs=params.discriminator_dim,
                activation_fn=tf.nn.leaky_relu,
                scope='mlp_{}'.format(i),
                weights_regularizer=weight_regularizer
            )
        h = slim.fully_connected(
            inputs=h,
            num_outputs=1,
            activation_fn=None,
            scope='mlp_output',
            weights_regularizer=weight_regularizer
        )
        h = tf.squeeze(h, axis=-1)
    return h
