import tensorflow as tf
from tensorflow.contrib import slim

from .lstm_utils import bilstm


def ctc_mm_network(
        utterances_t,
        utterance_lengths,
        vocab_size,
        params,
        utterance_idx,
        latent,
        is_training=True,
        weight_regularizer=None
):
    decoder_dropout = params.decoder_dropout
    with tf.variable_scope('bilstm'):
        h = utterances_t
        if latent is not None:
            h = tf.concat([h, latent], axis=-1)
        if params.skipconnect:
            for i in range(params.decoder_depth):
                with tf.variable_scope('decoder_layer_{}'.format(i)):
                    h, h_final = bilstm(
                        x=h,
                        num_layers=1,
                        num_units=params.decoder_dim,
                        sequence_lengths=utterance_lengths,
                        dropout=decoder_dropout,
                        is_training=is_training,
                        kernel_regularizer=weight_regularizer,
                        residual=params.residual
                    )
                    if params.batch_norm:
                        h = sequence_batch_norm(
                            inputs=h,
                            scope="output_bn_{}".format(i),
                            is_training=is_training,
                            idx=utterance_idx
                        )
                    if decoder_dropout > 0:
                        h = slim.dropout(h, keep_prob=1 - decoder_dropout, is_training=is_training)
                    if latent is not None:
                        h = tf.concat([h, latent], axis=-1)
        else:
            with tf.variable_scope('decoder'):
                h, h_final = bilstm(
                    x=h,
                    num_layers=params.decoder_depth,
                    num_units=params.decoder_dim,
                    sequence_lengths=utterance_lengths,
                    dropout=decoder_dropout,
                    is_training=is_training,
                    kernel_regularizer=weight_regularizer,
                    residual=params.residual
                )
                if params.batch_norm:
                    h = sequence_batch_norm(
                        inputs=h,
                        scope="output_bn",
                        is_training=is_training,
                        idx=utterance_idx
                    )
                if decoder_dropout > 0:
                    h = slim.dropout(h, keep_prob=1 - decoder_dropout, is_training=is_training)
        h = slim.fully_connected(
            h,
            num_outputs=vocab_size + 1,
            activation_fn=None,
            weights_regularizer=weight_regularizer,
            scope='output_unit')
        gate_logits = slim.fully_connected(
            h_final,
            num_outputs=1,
            activation_fn=None,
            weights_regularizer=weight_regularizer,
            scope='gate_logits')
    return h, gate_logits
