import tensorflow as tf
from tensorflow.contrib import slim

from .lstm_utils import bilstm


def gating_network(
        utterances_t,
        utterance_lengths,
        params,
        is_training=True,
        weight_regularizer=None
):
    decoder_dropout = params.decoder_dropout
    with tf.variable_scope('bilstm'):
        with tf.variable_scope('gating'):
            _, h = bilstm(
                x=utterances_t,
                num_layers=params.gating_depth,
                num_units=params.gating_dim,
                sequence_lengths=utterance_lengths,
                dropout=decoder_dropout,
                is_training=is_training,
                kernel_regularizer=weight_regularizer,
                residual=params.residual
            )
            h = slim.fully_connected(
                h,
                num_outputs=params.mm_size,
                activation_fn=None,
                weights_regularizer=weight_regularizer,
                scope='output_unit')
    return h
