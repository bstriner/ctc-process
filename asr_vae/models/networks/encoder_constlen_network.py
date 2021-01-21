import tensorflow as tf

from .fully_connected import fully_connected, fully_connected_layer
from .lstm_utils import bilstm
from .sampling import latent_sample_fn
from .variational.variational_variable import VariationalParams


def encoder_constlen_network(
        utterances_t,
        utterance_lengths,
        transcripts_t,
        transcript_lengths,
        params,
        is_training,
        vocab_size,
        vparams: VariationalParams):
    n = tf.shape(utterances_t)[1]
    assert not params.encoder_dropout > 0.
    assert params.batch_norm.lower() == 'none'
    with tf.variable_scope('utterances_bilstm'):
        _, utterance_hidden = bilstm(
            inputs=utterances_t,
            num_layers=params.encoder_depth,
            num_units=params.encoder_dim,
            sequence_lengths=None if params.constlen_lstm else utterance_lengths,
            is_training=is_training,
            vparams=vparams,
            residual=params.residual,
            dropout_fn=None,
            batch_norm_fn=None,
            rnn_mode=params.rnn_mode
        )  # (N,D)

    with tf.variable_scope('transcripts_bilstm'):
        embeddings = tf.get_variable(
            dtype=tf.float32,
            name="embeddings",
            shape=[vocab_size, params.encoder_dim],
            initializer=tf.initializers.truncated_normal(
                stddev=1. / tf.sqrt(tf.constant(params.encoder_dim, dtype=tf.float32))))
        transcripts_embedded = tf.nn.embedding_lookup(embeddings, transcripts_t)
        _, transcripts_hidden = bilstm(
            inputs=transcripts_embedded,
            num_layers=params.encoder_depth,
            num_units=params.encoder_dim,
            sequence_lengths=None if params.constlen_lstm else transcript_lengths,
            is_training=is_training,
            residual=params.residual,
            dropout_fn=None,
            batch_norm_fn=None,
            vparams=vparams,
            rnn_mode=params.rnn_mode
        )  # (N,D)
    with tf.variable_scope('encoder_network'):
        if params.model in ['vae', 'aae', 'ae']:
            encoder_inputs = tf.concat([utterance_hidden, transcripts_hidden], axis=-1)
        elif params.model == 'aae-stoch':
            noise = tf.random.normal(shape=(n, params.noise_dim))
            encoder_inputs = tf.concat([utterance_hidden, transcripts_hidden, noise], axis=-1)
        elif params.model == 'ctc':
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        h = encoder_inputs
        for i in range(params.encoder_depth):
            h = fully_connected(
                inputs=h,
                num_outputs=params.encoder_dim,
                activation_fn=tf.nn.leaky_relu,
                scope='mlp_{}'.format(i),
                residual=params.residual,
                vparams=vparams
            )
        if params.model == 'vae' or params.model == 'aae':
            mu = fully_connected_layer(
                inputs=h,
                num_outputs=params.latent_dim,
                activation_fn=None,
                scope='mu',
                vparams=vparams
            )
            logsigmasq = fully_connected_layer(
                inputs=h,
                num_outputs=params.latent_dim,
                activation_fn=None,
                scope='sigma',
                vparams=vparams
            )
            latent, latent_prior, metrics = latent_sample_fn(
                mu=mu,
                logsigmasq=logsigmasq,
                params=params,
                n=n
            )
        elif params.model == 'ae':
            latent = fully_connected_layer(
                inputs=h,
                num_outputs=params.latent_dim,
                activation_fn=None,
                scope='latent',
                vparams=vparams
            )
            latent_prior = latent
            metrics = {}
        elif params.model == 'aae-stoch':
            latent = fully_connected_layer(
                inputs=h,
                num_outputs=params.latent_dim,
                activation_fn=None,
                scope='latent',
                vparams=vparams
            )
            latent_prior = tf.random.normal(shape=tf.shape(latent))
            metrics = {}
        elif params.model == 'ctc':
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    return latent, latent_prior, metrics
