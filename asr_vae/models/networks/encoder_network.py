import tensorflow as tf
from tensorflow.contrib import slim

from .attention import attention_fn_softmax_loop
from .lstm_utils import bilstm
from .sampling import latent_sample_fn
from ..helpers.estimator import SLOW_SUMMARIES


def encoder_network(
        utterances_t,
        utterance_lengths,
        transcripts_t,
        transcript_lengths,
        params,
        is_training,
        vocab_size,
        utterance_idx,
        vparams,
        transcript_idx,
        weight_regularizer=None):
    n = tf.shape(utterances_t)[1]
    ul = tf.shape(utterances_t)[0]
    tl = tf.shape(transcripts_t)[0]
    assert not params.encoder_dropout > 0.
    assert params.batch_norm.lower() == 'none'
    with tf.variable_scope('utterances_bilstm'):
        utterance_hidden, _ = bilstm(
            inputs=utterances_t,
            num_layers=params.encoder_depth,
            num_units=params.encoder_dim,
            sequence_lengths=None if params.constlen_lstm else utterance_lengths,
            dropout_fn=None,
            batch_norm_fn=None,
            is_training=is_training,
            residual=params.residual,
            vparams=vparams,
            rnn_mode=params.rnn_mode
        )
    with tf.variable_scope('transcripts_bilstm'):
        embeddings = tf.get_variable(
            dtype=tf.float32,
            name="embeddings",
            shape=[vocab_size, params.encoder_dim],
            initializer=tf.initializers.truncated_normal(
                stddev=1. / tf.sqrt(tf.constant(params.encoder_dim, dtype=tf.float32))),
            regularizer=weight_regularizer)
        transcripts_embedded = tf.nn.embedding_lookup(embeddings, transcripts_t)
        transcripts_hidden, _ = bilstm(
            inputs=transcripts_embedded,
            num_layers=params.encoder_depth,
            num_units=params.encoder_dim,
            sequence_lengths=None if params.constlen_lstm else transcript_lengths,
            dropout_fn=None,
            batch_norm_fn=None,
            is_training=is_training,
            residual=params.residual,
            vparams=vparams,
            rnn_mode=params.rnn_mode
        )
    with tf.variable_scope('attention'):
        # utterance_mask = tf.transpose(tf.sequence_mask(maxlen=ul, lengths=utterance_lengths))
        # transcript_mask = tf.transpose(tf.sequence_mask(maxlen=tl, lengths=transcript_lengths))
        attn = attention_fn_softmax_loop(
            utterance_h=utterance_hidden,
            transcript_h=transcripts_hidden,
            dim=params.attention_dim,
            # utterance_mask=utterance_mask,
            # transcript_mask=transcript_mask
            utterance_lengths=utterance_lengths,
            transcript_lengths=transcript_lengths,
            weight_regularizer=weight_regularizer,
            batch_first=False
        )  # (n, ul, tl)

    tf.summary.image(
        "Attention",
        tf.expand_dims(attn, 3),
        collections=[SLOW_SUMMARIES])

    transcripts_data = tf.concat([transcripts_embedded, transcripts_hidden], axis=-1)
    aligned_transcripts_hidden = tf.matmul(
        attn,  # (n, ul, tl)
        tf.transpose(transcripts_data, (1, 0, 2))  # (n, tl, d)
    )  # (n, ul, d)
    aligned_transcripts_hidden = tf.transpose(aligned_transcripts_hidden, (1, 0, 2))  # (ul, n, d)
    # tmp1 = tf.expand_dims(tf.transpose(attn, (0, 2, 1)), 2)  # (ul, n, 1, tl)
    # tmp2 = tf.expand_dims(tf.transpose(transcript_hid, (1, 2, 0)), 0)  # (1, n, d, tl)
    # aligned_transcript_h = tf.reduce_sum(tmp1 * tmp2, axis=3)  # (u1, n, d)
    if params.model == 'ctc':
        raise NotImplementedError()
    elif params.model in ['aae', 'vae', 'ae']:
        encoder_inputs = tf.concat((utterances_t, utterance_hidden, aligned_transcripts_hidden), axis=-1)
    elif params.model == 'aae-stoch':
        noise = tf.random.normal(
            shape=(n, ul, params.noise_dim)
        )
        encoder_inputs = tf.concat((utterances_t, utterance_hidden, aligned_transcripts_hidden, noise), axis=-1)
    else:
        raise NotImplementedError()

    with tf.variable_scope('encoder_output'):
        h = encoder_inputs
        for i in range(params.encoder_depth):
            with tf.variable_scope('encoder_output_{}'.format(i)):
                h, _ = bilstm(
                    inputs=h,
                    num_layers=params.encoder_depth,
                    num_units=params.encoder_dim,
                    sequence_lengths=None if params.constlen_lstm else utterance_lengths,
                    dropout_fn=None,
                    batch_norm_fn=None,
                    vparams=vparams,
                    is_training=is_training,
                    residual=params.residual,
                    rnn_mode=params.rnn_mode
                )
                h = tf.concat([h, encoder_inputs], axis=-1)

        if params.model == 'ctc':
            raise NotImplementedError()
        elif params.model == 'ae':
            latent = slim.fully_connected(
                h,
                num_outputs=params.latent_dim,
                activation_fn=None,
                weights_regularizer=weight_regularizer,
                scope='latent')
            latent_prior = latent
            metrics = {}
        elif params.model == 'aae' or params.model == 'vae':
            encoder_output_masked = tf.gather_nd(params=h, indices=utterance_idx)
            mu_masked = slim.fully_connected(
                encoder_output_masked,
                num_outputs=params.latent_dim,
                activation_fn=None,
                weights_regularizer=weight_regularizer,
                scope='mu')
            logsigmasq_masked = slim.fully_connected(
                encoder_output_masked,
                num_outputs=params.latent_dim,
                activation_fn=None,
                weights_regularizer=weight_regularizer,
                scope='sigma')
            latent_masked, latent_prior_masked, metrics = latent_sample_fn(
                mu=mu_masked,
                logsigmasq=logsigmasq_masked,
                params=params,
                n=n
            )
            latent = tf.scatter_nd(
                updates=latent_masked,
                shape=(ul, n, params.latent_dim),
                indices=utterance_idx)
            latent_prior = tf.scatter_nd(
                updates=latent_prior_masked,
                shape=(ul, n, params.latent_dim),
                indices=utterance_idx)
        elif params.model == 'aae-stoch':
            latent = slim.fully_connected(
                h,
                num_outputs=params.latent_dim,
                activation_fn=None,
                weights_regularizer=weight_regularizer,
                scope='latent')
            latent_prior = tf.random.normal(
                shape=tf.shape(latent)
            )
            metrics = {}
        else:
            raise NotImplementedError()

    return latent, latent_prior, metrics
