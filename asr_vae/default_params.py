import json
import os

import six
import tensorflow as tf
from tensorflow.contrib.training import HParams

HPARAMS = 'configuration-hparams.json'


def load_hparams(model_dir):
    hparams = default_params()
    hparams_path = os.path.join(model_dir, HPARAMS)
    assert os.path.exists(hparams_path)

    with open(hparams_path) as f:
        return hparams.parse_json(f.read())


def get_hparams(model_dir, validate=True):
    config_file = tf.app.flags.FLAGS.config
    hparams = default_params()
    hparams_path = os.path.join(model_dir, HPARAMS)
    with open(config_file) as f:
        hparams.parse_json(f.read())
    hparams_str = tf.app.flags.FLAGS.hparams
    hparams.parse(hparams_str)

    if os.path.exists(hparams_path) and validate:
        with open(hparams_path) as f:
            hparam_dict = json.load(f)
        for k, v in six.iteritems(hparam_dict):
            oldval = getattr(hparams, k)
            assert oldval == v, "Incompatible key {}: save {}-> config {}".format(k, oldval, v)
    with open(hparams_path, 'w') as f:
        json.dump(obj=hparams.values(), fp=f, sort_keys=True, indent=4)
    return hparams


def default_params():
    return HParams(
        train_acc=1,

        model='ctc',
        aae_mode='gan',
        residual=False,
        batch_norm='none',
        batch_norm_scale=True,
        input_scale=1.0,
        optimizer='adadelta',

        variational_mode="none",
        variational_sigma_init=0.075,
        variational_sigma_prior=1.0,
        variational_scale=1.0 / 37416.0,

        decoder_dim=320,
        decoder_depth=4,
        decoder_pyramid_depth=0,
        decoder_dropout=0.,
        decoder_uout=False,
        encoder_dim=320,
        encoder_depth=5,
        encoder_dropout=0.,
        discriminator_dim=320,
        discriminator_depth=5,

        subsample=3,
        independent_subsample=True,
        lr=1.0,
        dis_lr=3e-4,
        l2=0.,

        attention_dim=128,
        latent_dim=128,

        anneal_start=2000,
        anneal_end=10000,
        anneal_min=1e-3,
        anneal_max=1.0,
        kl_min=1e-2,
        clip_gradient_norm=0.,
        clip_gradient_value=0.1,

        gating_depth=3,
        gating_dim=256,
        mm_size=4,

        penalty_weight=10.,
        discriminator_steps=5,
        anneal_scale='log',
        logsigmasq_min=-100.0,
        logsigmasq_max=10.0,
        logsigmasq_clip=False,
        mu_clip=0.0,
        logit_clip=0.0,
        latent_tile_scale=1.0,
        encoder_lr=3e-4,
        momentum=0.9,
        cudnn=True,
        constlen_lstm=False,
        flat_latent=True,
        bucket=True,
        bucket_size=100,
        buckets=30,
        epochs_without_improvement=100,
        lr_scale=False,
        lr_rate=0.5,
        lr_min=0.00001,
        specaugment=False,
        specaugment_W=5,
        specaugment_T=40,
        specaugment_F=30,
        specaugment_mT=2,
        specaugment_mF=2,
        specaugment_p=1.0,
        rnn_mode="lstm",
        preproc_network='none',
        postproc_network='none',
        sentencepiece_online=False
    )
