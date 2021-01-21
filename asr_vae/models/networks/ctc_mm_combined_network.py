import tensorflow as tf

from .dropout import make_dropout_fn
from .fully_connected import fully_connected_layer
from .lstm_utils import bilstm
from .normalization import make_sequence_batch_normalization_fn


def ctc_mm_combined_network(
        utterances_t,
        utterance_lengths,
        vocab_size,
        params,
        vparams,
        is_training=True
):
    l = tf.shape(utterances_t)[0]
    mm_size = params.mm_size
    with tf.variable_scope('decoder'):
        batch_norm_fn = make_sequence_batch_normalization_fn(
            is_training=is_training,
            params=params,
            sequence_lengths=utterance_lengths,
            maxlen=l
        )
        dropout_fn = make_dropout_fn(
            is_training=is_training,
            dropout=params.decoder_dropout,
            uout=params.decoder_uout
        )
        h, hfinal = bilstm(
            inputs=utterances_t,
            num_layers=params.decoder_depth,
            num_units=params.decoder_dim,
            sequence_lengths=None if params.constlen_lstm else utterance_lengths,
            is_training=is_training,
            residual=params.residual,
            dropout_fn=dropout_fn,
            vparams=vparams,
            batch_norm_fn=batch_norm_fn,
            cudnn=params.cudnn,
            rnn_mode=params.rnn_mode
        )
        logits = []
        for i in range(mm_size):
            logit = fully_connected_layer(
                inputs=h,
                num_outputs=vocab_size + 1,
                activation_fn=None,
                batch_norm_fn=None,
                vparams=vparams,
                scope='output_unit_{}'.format(i))
            logits.append(logit)
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
        return logits, mixture_logits
