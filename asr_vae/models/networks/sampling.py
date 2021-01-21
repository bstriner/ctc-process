import tensorflow as tf

from ..metrics.asr_metrics import EVAL_SUMMARIES, SLOW_SUMMARIES
from ...anneal import get_scale

# import tensorflow_probability as tfp
# tfd = tfp.distributions

"""
def kl_loss_fn(latent_dist, latent_prior, params, n):
    if params.model == 'vae':
        kl_n = tfd.kl_divergence(latent_dist, latent_prior)
        kl_n = tf.maximum(kl_n, params.kl_min)
        print("kl_n shape: {}".format(kl_n))
        kl_loss_raw = tf.reduce_sum(kl_n) / tf.cast(n, tf.float32)
        kl_scale = get_scale(params)
        kl_loss = kl_loss_raw * kl_scale
        kl_loss = tf.check_numerics(kl_loss, "kl_loss", name='kl_loss_check_numerics')

        tf.summary.scalar("kl_loss_raw", kl_loss_raw)
        tf.summary.scalar("kl_scale", kl_scale)
        tf.summary.scalar("kl_loss_scaled", kl_loss)
        tf.losses.add_loss(kl_loss)
"""


def logsigmasq_loss_fn(logsigmasq):
    return -0.5 * (1 + logsigmasq - tf.exp(logsigmasq))


def mu_loss_fn(mu):
    return -0.5 * (- tf.square(mu))

def mu_regularizer_loss_fn(mu):
    return tf.reduce_sum(mu_loss_fn(mu=mu))


def logsigmasq_regularizer_loss_fn(logsigmasq):
    return tf.reduce_sum(logsigmasq_loss_fn(logsigmasq=logsigmasq))

def draw_sample(mu, logsigmasq):
    noise = tf.random.normal(shape=tf.shape(mu))
    sigma = tf.exp(logsigmasq / 2.)
    latent_sample = mu + (noise * sigma)
    return latent_sample


def differentiable_clip(x, clip_value_min, clip_value_max):
    clipped = tf.clip_by_value(x, clip_value_min=clip_value_min, clip_value_max=clip_value_max)
    return tf.stop_gradient(clipped - x) + x


def kl_loss_fn(mu, logsigmasq, params, n):
    if params.model == 'vae':
        kl_n = -0.5 * (1 + logsigmasq - tf.square(mu) - tf.exp(logsigmasq))
        kl_n = tf.maximum(kl_n, params.kl_min)
        kl_loss_raw = tf.reduce_sum(kl_n) / tf.cast(n, tf.float32)
        kl_scale = get_scale(params)
        kl_loss = kl_loss_raw * kl_scale
        kl_loss = tf.check_numerics(kl_loss, "kl_loss", name='kl_loss_check_numerics')

        tf.summary.scalar("kl_loss_raw", kl_loss_raw)
        tf.summary.scalar("kl_scale", kl_scale)
        tf.summary.scalar("kl_loss_scaled", kl_loss)
        tf.losses.add_loss(kl_loss)
        ns = tf.get_default_graph().get_name_scope()
        metrics = {
            "{}/kl_loss_raw".format(ns): tf.metrics.mean(kl_loss_raw),
            "{}/kl_loss_scaled".format(ns): tf.metrics.mean(kl_loss)
        }
    else:
        metrics = {}
    return metrics


def latent_sample_fn(mu, logsigmasq, params, n):
    # logsigmasq = tf.check_numerics(logsigmasq, 'logsigmasq check_numerics')
    # mu = tf.check_numerics(mu, 'mu check_numerics')
    """
    logsigmasq = differentiable_clip(
        logsigmasq,
        clip_value_min=params.logsigmasq_min,
        clip_value_max=params.logsigmasq_max)
    if params.logsigmasq_clip:
        logsigmasq_clip = tf.clip_by_value(
            logsigmasq,
            clip_value_min=params.logsigmasq_min,
            clip_value_max=params.logsigmasq_max)
    else:
        logsigmasq_clip = logsigmasq
    if params.mu_clip > 0:
        # mu = differentiable_clip(
        #    mu,
        #    clip_value_min=-params.mu_clip,
        #    clip_value_max=params.mu_clip)
        mu_clip = tf.clip_by_value(mu, clip_value_min=-params.mu_clip, clip_value_max=params.mu_clip)
    else:
        mu_clip = mu
    """
    latent_sample = draw_sample(mu=mu, logsigmasq=logsigmasq)
    latent_prior_sample = tf.random.normal(shape=tf.shape(mu))

    metrics = kl_loss_fn(
        mu=mu,
        logsigmasq=logsigmasq,
        params=params,
        n=n
    )
    tf.summary.histogram('mu', mu, collections=[EVAL_SUMMARIES, SLOW_SUMMARIES])
    tf.summary.histogram('logsigmasq', logsigmasq, collections=[EVAL_SUMMARIES, SLOW_SUMMARIES])
    return latent_sample, latent_prior_sample, metrics
