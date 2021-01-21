import tensorflow as tf

from .dropout import make_dropout_fn
from .fully_connected import fully_connected_layer
from .lstm_utils import bilstm, sequence_pyramid, sequence_pyramid_lengths
from .normalization import make_sequence_batch_normalization_fn
from .postproc_cnn1d import postproc_cnn1d
from .variational.variational_variable import VariationalParams


def postproc(inputs, input_lengths, params, is_training=True):
    if params.postproc_network == 'none':
        return inputs, input_lengths
    elif params.postproc_network == 'conv1d_1layer_s2':
        return postproc_cnn1d(
            inputs=inputs,
            params=params,
            is_training=is_training,
            strides=[2]
        )
    elif params.postproc_network == 'conv1d_2layer_s4':
        return postproc_cnn1d(
            inputs=inputs,
            params=params,
            is_training=is_training,
            strides=[2, 2]
        )
    else:
        raise ValueError()


def ctc_network(
        utterances_t,
        utterance_lengths,
        vocab_size,
        params,
        vparams: VariationalParams,
        latent,
        is_training=True,
        mm=False
):
    if mm:
        mm_size = params.mm_size
    else:
        mm_size = 1
    dropout_fn = make_dropout_fn(
        is_training=is_training,
        dropout=params.decoder_dropout,
        uout=params.decoder_uout
    )
    with tf.variable_scope('bilstm'):
        h = utterances_t
        if latent is not None:
            h = tf.concat([h, latent], axis=-1)
        with tf.variable_scope('decoder_layers'):
            for i in range(params.decoder_pyramid_depth):
                with tf.variable_scope('decoder_layer_pyramid_{}'.format(i)):
                    h, _ = bilstm(
                        inputs=h,
                        num_layers=1,
                        num_units=params.decoder_dim,
                        sequence_lengths=None if params.constlen_lstm else utterance_lengths,
                        is_training=is_training,
                        residual=params.residual,
                        dropout_fn=dropout_fn,
                        vparams=vparams,
                        batch_norm_fn=make_sequence_batch_normalization_fn(
                            is_training=is_training,
                            params=params,
                            sequence_lengths=utterance_lengths,
                            maxlen=tf.shape(h)[0]
                        ),
                        rnn_mode=params.rnn_mode
                    )
                    h = sequence_pyramid(inputs=h)
                    utterance_lengths = sequence_pyramid_lengths(sequence_lengths=utterance_lengths)
            batch_norm_fn = make_sequence_batch_normalization_fn(
                is_training=is_training,
                params=params,
                sequence_lengths=utterance_lengths,
                maxlen=tf.shape(h)[0]
            )
            h, hfinal = bilstm(
                inputs=h,
                num_layers=params.decoder_depth,
                num_units=params.decoder_dim,
                sequence_lengths=None if params.constlen_lstm else utterance_lengths,
                is_training=is_training,
                residual=params.residual,
                dropout_fn=dropout_fn,
                vparams=vparams,
                batch_norm_fn=batch_norm_fn,
                rnn_mode=params.rnn_mode
            )
            h, utterance_lengths = postproc(
                inputs=h,
                input_lengths=utterance_lengths,
                params=params,
                is_training=is_training
            )
            logits = fully_connected_layer(
                inputs=h,
                num_outputs=mm_size * (vocab_size + 1),
                activation_fn=None,
                batch_norm_fn=None,
                vparams=vparams,
                scope='output_unit')
            if mm:
                n = tf.shape(logits)[0]
                l = tf.shape(logits)[1]
                logits = tf.reshape(logits, [n, l, mm_size, vocab_size + 1])
                with tf.variable_scope("mixture_logit_network"):
                    if batch_norm_fn is not None:
                        hfinal = batch_norm_fn(hfinal)
                    if dropout_fn is not None:
                        hfinal = dropout_fn(hfinal)
                    mixture_logits = fully_connected_layer(
                        inputs=hfinal,
                        num_outputs=mm_size,
                        activation_fn=None,
                        batch_norm_fn=None,
                        vparams=vparams,
                        scope='mixture_logits')
                return logits, utterance_lengths, mixture_logits
            else:
                return logits, utterance_lengths, None
