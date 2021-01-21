import tensorflow as tf

from .aae import aae_losses_and_hooks
from .helpers.estimator import EVAL_SUMMARIES, SLOW_SUMMARIES, make_estimator
from .helpers.estimator_mm import make_estimator_mm
from .networks.ctc_network import ctc_network
from .networks.discriminator_constlen_network import discriminator_constlen_network
from .networks.discriminator_varlen_network import discriminator_varlen_network
from .networks.encoder_constlen_network import encoder_constlen_network
from .networks.encoder_network import encoder_network
from .networks.lstm_utils import sequence_index
from .networks.preproc_cnn1d_1layer import preproc_cnn1d_1layer
from .networks.preproc_cnn2d_2layer import preproc_cnn2d_2layer
from .networks.preproc_cnn2d_3layer import preproc_cnn2d_3layer
from .networks.variational.variational_variable import VARIATIONAL_LOSSES, VariationalParams
from ..kaldi.constants import *


def preproc(utterances, utterance_lengths, params, is_training):
    if params.preproc_network == 'none':
        return utterances, utterance_lengths
    elif params.preproc_network == 'cnn2d_3layer':
        return preproc_cnn2d_3layer(
            inputs=utterances,
            params=params,
            is_training=is_training,
            strides=[(2, 2), (1, 2), (1, 2)]
        )
    elif params.preproc_network == 'cnn2d_3layer_s4':
        return preproc_cnn2d_3layer(
            inputs=utterances,
            params=params,
            is_training=is_training,
            strides=[(2, 2), (2, 2), (1, 2)]
        )
    elif params.preproc_network == 'cnn2d_3layer_s8':
        return preproc_cnn2d_3layer(
            inputs=utterances,
            params=params,
            is_training=is_training,
            strides=[(2, 2), (2, 2), (2, 2)]
        )
    elif params.preproc_network == 'cnn2d_2layer':
        return preproc_cnn2d_2layer(
            inputs=utterances,
            params=params,
            is_training=is_training,
            strides=[(2, 2), (1, 2)]
        )
    elif params.preproc_network == 'cnn2d_2layer_s4':
        return preproc_cnn2d_2layer(
            inputs=utterances,
            params=params,
            is_training=is_training,
            strides=[(2, 2), (2, 2)]
        )
    elif params.preproc_network == 'cnn1d_1layer':
        return preproc_cnn1d_1layer(
            inputs=utterances,
            params=params,
            is_training=is_training
        )
    else:
        raise ValueError()


def make_model_fn(
        run_config,
        vocab,
        sentencepiece
):
    def model_fn(features, labels, mode, params):
        if params.model == 'ctc-mm':
            mm = True
        else:
            mm = False

        vparams = VariationalParams(
            mode=params.variational_mode,
            sigma_init=params.variational_sigma_init,
            mu_prior=0.0,
            sigma_prior=params.variational_sigma_prior,
            scale=params.variational_scale
        )

        is_training = mode == tf.estimator.ModeKeys.TRAIN
        utterances, utterance_lengths = features[FEATURES], features[FEATURE_LENGTH]  # (N, L, 64)
        transcripts, transcript_lengths = features[LABELS], features[LABEL_LENGTH]  # (N, T)
        #tf.summary.image("mel_input", tf.expand_dims(utterances, 3), collections=[SLOW_SUMMARIES, EVAL_SUMMARIES])
        transcripts_t = tf.transpose(transcripts, (1, 0))  # (u, n, dim)
        vocab_size = vocab.shape[0]
        n = tf.shape(utterances)[0]
        tl = tf.shape(transcripts)[1]

        transcript_idx = sequence_index(
            lengths=transcript_lengths,
            maxlen=tl
        )
        print("Vocab size: {}".format(vocab_size))
        """
        if params.l2 > 0 and params.optimizer != 'adamw':
            def weight_regularizer(x):
                return params.l2 * tf.reduce_sum(tf.square(x))
        else:
            weight_regularizer = None
        """
        with tf.variable_scope("ctc_model") as model_scope:
            utterances, utterance_lengths = preproc(
                utterances=utterances,
                utterance_lengths=utterance_lengths,
                params=params,
                is_training=is_training
            )
            utterances_t = tf.transpose(utterances, (1, 0, 2))  # (u, n, dim)
            ul = tf.shape(utterances)[1]
            utterance_idx = sequence_index(
                lengths=utterance_lengths,
                maxlen=ul
            )
            if params.model in ['ctc', 'ctc-mm']:
                latent, latent_prior = None, None
                latent_tiled = None
                glatent_tiled = None
                metrics = {}
                encoder_scope = None
            elif params.model in ['vae', 'aae', 'aae-stoch', 'ae']:
                with tf.variable_scope("encoder_network", reuse=False) as encoder_scope:
                    if params.flat_latent:
                        latent, latent_prior, metrics = encoder_constlen_network(
                            utterances_t=utterances_t,
                            utterance_lengths=utterance_lengths,
                            transcripts_t=transcripts_t,
                            transcript_lengths=transcript_lengths,
                            params=params,
                            is_training=is_training,
                            vocab_size=vocab_size,
                            # weight_regularizer=weight_regularizer,
                            vparams=vparams
                        )
                        latent_tiled = tf.tile(tf.expand_dims(latent, 0), [ul, 1, 1]) * params.latent_tile_scale
                        glatent_tiled = tf.tile(tf.expand_dims(latent_prior, 0), [ul, 1, 1]) * params.latent_tile_scale
                        # decoder_inputs = tf.concat([utterances_t, latent_tiled], axis=-1)
                        # gdecoder_inputs = tf.concat([utterances_t, glatent_tiled], axis=-1)
                    else:
                        latent, latent_prior, metrics = encoder_network(
                            utterances_t=utterances_t,
                            utterance_lengths=utterance_lengths,
                            transcripts_t=transcripts_t,
                            transcript_lengths=transcript_lengths,
                            params=params,
                            is_training=is_training,
                            vocab_size=vocab_size,
                            # weight_regularizer=weight_regularizer,
                            utterance_idx=utterance_idx,
                            transcript_idx=transcript_idx,
                            vparams=vparams
                        )
                        latent_tiled = latent
                        glatent_tiled = latent_prior
                        # decoder_inputs = tf.concat([utterances_t, latent], axis=-1)
                        # gdecoder_inputs = tf.concat([utterances_t, latent_prior], axis=-1)
            else:
                raise NotImplementedError()

            with tf.variable_scope("decoder_network", reuse=False) as decoder_scope:
                with tf.name_scope(decoder_scope.name + "/autoencoded/"):
                    logits_t, logit_lengths, gate_logits = ctc_network(
                        utterances_t=utterances_t,
                        utterance_lengths=utterance_lengths,
                        vocab_size=vocab_size,
                        is_training=is_training,
                        params=params,
                        latent=latent_tiled,
                        vparams=vparams,
                        mm=mm)
            if params.model in ['ctc', 'ctc-mm']:
                glogits_t = logits_t
            elif params.model in ['vae', 'aae', 'aae-stoch', 'ae']:
                with tf.variable_scope(decoder_scope, reuse=True):
                    with tf.name_scope(decoder_scope.name + "/generated/"):
                        glogits_t, _, _ = ctc_network(
                            utterances_t=utterances_t,
                            utterance_lengths=utterance_lengths,
                            vocab_size=vocab_size,
                            is_training=is_training,
                            params=params,
                            latent=glatent_tiled,
                            vparams=vparams,
                            mm=mm)
            else:
                raise ValueError()
            if vparams.enabled:
                vloss = tf.add_n(tf.get_collection(
                    key=VARIATIONAL_LOSSES,
                    scope=model_scope.name
                ))
                tf.summary.scalar('variational_losses', vloss, collections=[tf.GraphKeys.SUMMARIES, EVAL_SUMMARIES])

        if params.model in ['aae', 'aae-stoch']:
            if is_training:
                if params.constlen:
                    def discriminator_fn(input_latent):
                        return discriminator_constlen_network(
                            utterances_t=utterances_t,
                            utterance_lengths=utterance_lengths,
                            latent=input_latent,
                            params=params,
                            is_training=is_training,
                            # weight_regularizer=weight_regularizer
                        )
                else:
                    def discriminator_fn(input_latent):
                        return discriminator_varlen_network(
                            utterances_t=utterances_t,
                            utterance_lengths=utterance_lengths,
                            latent=input_latent,
                            params=params,
                            is_training=is_training,
                            # weight_regularizer=weight_regularizer
                        )
                dis_train_hook = aae_losses_and_hooks(
                    real=latent_prior,
                    fake=latent,
                    params=params,
                    discriminator_fn=discriminator_fn,
                    model_scope=model_scope.name,
                    axis=0
                )
                training_hooks = [dis_train_hook]
            else:
                training_hooks = []
        elif params.model in ['vae', 'ctc', 'ae', 'ctc-mm']:
            training_hooks = []
        else:
            raise NotImplementedError()

        if mm:
            return make_estimator_mm(
                params=params,
                logits_t=logits_t,
                gate_logits=gate_logits,
                logit_lengths=logit_lengths,
                transcripts=transcripts,
                transcript_lengths=transcript_lengths,
                mode=mode,
                vocab=vocab,
                run_config=run_config,
                model_scope=model_scope.name,
                is_training=is_training,
                training_hooks=training_hooks,
                encoder_scope=encoder_scope,
                decoder_scope=decoder_scope,
                utterance_id=features[UTTERANCE_ID],
                eval_metric_ops=metrics,
                sentencepiece=sentencepiece)
        else:
            return make_estimator(
                params=params,
                logits_t=logits_t,
                glogits_t=glogits_t,
                logit_lengths=logit_lengths,
                transcripts=transcripts,
                transcript_lengths=transcript_lengths,
                mode=mode,
                vocab=vocab,
                run_config=run_config,
                model_scope=model_scope.name,
                is_training=is_training,
                training_hooks=training_hooks,
                encoder_scope=encoder_scope,
                decoder_scope=decoder_scope,
                utterance_id=features[UTTERANCE_ID],
                eval_metric_ops=metrics,
                sentencepiece=sentencepiece)

    return model_fn
