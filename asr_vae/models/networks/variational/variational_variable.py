import numpy as np
import tensorflow as tf

from ..sampling import draw_sample

VARIATIONAL_LOSSES = 'VARIATIONAL_LOSSES'


class VariationalParams(object):
    def __init__(
            self,
            mode,
            sigma_init,
            mu_prior,
            sigma_prior,
            scale
    ):
        self.mode = mode
        self.enabled = mode != 'none'
        self.sigma_init = sigma_init
        self.mu_prior = mu_prior
        self.sigma_prior = sigma_prior
        self.scale = scale


# https://www.cs.toronto.edu/~graves/nips_2011.pdf
def make_mu_regularizer_loss_fn(mu_prior, logsigmasq_prior, variational_scale):
    def mu_regularizer_loss_fn(mu):
        loss = 0.5 * tf.reduce_sum(
            (tf.square(mu - mu_prior) / tf.exp(logsigmasq_prior))
        ) * variational_scale
        tf.add_to_collection(
            name=VARIATIONAL_LOSSES,
            value=loss
        )
        return loss

    return mu_regularizer_loss_fn


def make_logsigmasq_regularizer_loss_fn(logsigmasq_prior, variational_scale):
    def logsigmasq_regularizer_loss_fn(logsigmasq):
        loss = 0.5 * tf.reduce_sum(
            logsigmasq_prior
            - logsigmasq
            + tf.exp(logsigmasq - logsigmasq_prior)
            - 1.0) * variational_scale
        tf.add_to_collection(
            name=VARIATIONAL_LOSSES,
            value=loss
        )
        return loss

    return logsigmasq_regularizer_loss_fn


def optimal_mu(mu):
    return tf.reduce_mean(mu)


def optimal_logsigmasq(mu, mu_prior, logsigmasq):
    """
    w = tf.reduce_prod(tf.cast(tf.shape(logsigmasq), tf.float32))
    p1 = tf.reduce_logsumexp(logsigmasq)-tf.log(w)
    p2 = tf.log(tf.reduce_mean(tf.squared_difference(mu, mu_prior)))
    p3 = tf.reduce_logsumexp(tf.stack([p1,p2], axis=-1))
    return p3
    """
    return tf.log(tf.reduce_mean(
        tf.exp(logsigmasq) + tf.squared_difference(mu, mu_prior)
    ))


def make_optimal_logsigmasq_regularizer_fn(mu, variational_scale, stopgrad=False):
    def regularizer_fn(logsigmasq):
        mu_prior = optimal_mu(
            mu=mu)
        logsigmasq_prior = optimal_logsigmasq(
            mu=mu,
            mu_prior=mu_prior,
            logsigmasq=logsigmasq)
        if stopgrad:
            mu_prior = tf.stop_gradient(mu_prior)
            logsigmasq_prior = tf.stop_gradient(logsigmasq_prior)
        mu_loss_fn = make_mu_regularizer_loss_fn(
            mu_prior=mu_prior,
            logsigmasq_prior=logsigmasq_prior,
            variational_scale=variational_scale
        )
        logsigmasq_loss_fn = make_logsigmasq_regularizer_loss_fn(
            logsigmasq_prior=logsigmasq_prior,
            variational_scale=variational_scale
        )
        loss = mu_loss_fn(mu) + logsigmasq_loss_fn(logsigmasq)
        return loss

    return regularizer_fn


def calc_logsigmasq(sigma):
    logsigmasq = 2.0 * np.log(sigma)
    return logsigmasq.astype(np.float32)


def variational_variable_static_prior(
        shape,
        vparams: VariationalParams, dtype=tf.float32,
        initializer=None, learn_prior=False
):
    if learn_prior:
        mu_prior = tf.get_variable(
            name="mu_prior",
            shape=[],
            dtype=dtype,
            initializer=tf.initializers.constant(vparams.mu_prior)
        )
        logsigmasq_prior = tf.get_variable(
            name="logsigmasq_prior",
            shape=[],
            dtype=dtype,
            initializer=tf.initializers.constant(calc_logsigmasq(vparams.sigma_prior))
        )
    else:
        mu_prior = tf.constant(vparams.mu_prior, dtype=dtype)
        logsigmasq_prior = tf.constant(calc_logsigmasq(vparams.sigma_prior), dtype=dtype)

    initializing_from_value = initializer is not None and not callable(initializer)
    if initializing_from_value:
        mu_shape = None
        assert shape == initializer.shape
    else:
        mu_shape = shape
    mu_regularizer = make_mu_regularizer_loss_fn(
        mu_prior=mu_prior,
        logsigmasq_prior=logsigmasq_prior,
        variational_scale=vparams.scale)
    mu = tf.get_variable(
        name='mu',
        shape=mu_shape,
        dtype=dtype,
        regularizer=mu_regularizer,
        initializer=initializer
    )

    logsigmasq_regularizer = make_logsigmasq_regularizer_loss_fn(
        logsigmasq_prior=logsigmasq_prior,
        variational_scale=vparams.scale)
    logsigmasq_init = calc_logsigmasq(vparams.sigma_init)
    logsigmasq_init = tf.initializers.constant(logsigmasq_init)
    logsigmasq = tf.get_variable(
        name='logsigmasq',
        shape=shape,
        dtype=dtype,
        regularizer=logsigmasq_regularizer,
        initializer=logsigmasq_init
    )
    tf.summary.scalar('mu_prior_value', mu_prior)
    tf.summary.scalar('logsigmasq_prior_value', logsigmasq_prior)
    return mu, logsigmasq


def variational_variable_adaptive_prior(
        shape,
        vparams: VariationalParams, dtype=tf.float32,
        initializer=None, stopgrad=False
):
    initializing_from_value = initializer is not None and not callable(initializer)
    if initializing_from_value:
        mu_shape = None
        assert shape == initializer.shape
    else:
        mu_shape = shape
    mu_regularizer = None
    mu = tf.get_variable(
        name='mu',
        shape=mu_shape,
        dtype=dtype,
        regularizer=mu_regularizer,
        initializer=initializer
    )

    logsigmasq_regularizer = make_optimal_logsigmasq_regularizer_fn(
        mu=mu,
        variational_scale=vparams.scale,
        stopgrad=stopgrad
    )
    logsigmasq_init = calc_logsigmasq(vparams.sigma_init)
    logsigmasq_init = tf.initializers.constant(logsigmasq_init)
    logsigmasq = tf.get_variable(
        name='logsigmasq',
        shape=shape,
        dtype=dtype,
        regularizer=logsigmasq_regularizer,
        initializer=logsigmasq_init
    )

    mu_prior = optimal_mu(
        mu=mu
    )
    logsigmasq_prior = optimal_logsigmasq(
        mu=mu,
        logsigmasq=logsigmasq,
        mu_prior=mu_prior
    )

    tf.summary.scalar('mu_prior_value', mu_prior)
    tf.summary.scalar('logsigmasq_prior_value', logsigmasq_prior)
    return mu, logsigmasq


def variational_variable(
        shape, name,
        vparams: VariationalParams, dtype=tf.float32,
        is_training=True, initializer=None
):
    with tf.variable_scope(name):
        if vparams.mode == 'adaptive':
            mu, logsigmasq = variational_variable_adaptive_prior(
                shape=shape,
                vparams=vparams,
                dtype=dtype,
                initializer=initializer,
                stopgrad=False)
        elif vparams.mode == 'adaptive-stopgrad':
            mu, logsigmasq = variational_variable_adaptive_prior(
                shape=shape,
                vparams=vparams,
                dtype=dtype,
                initializer=initializer,
                stopgrad=True)
        elif vparams.mode == 'adaptive-gd':
            mu, logsigmasq = variational_variable_static_prior(
                shape=shape,
                vparams=vparams,
                dtype=dtype,
                initializer=initializer,
                learn_prior=True)
        elif vparams.mode == 'static':
            mu, logsigmasq = variational_variable_static_prior(
                shape=shape,
                vparams=vparams,
                dtype=dtype,
                initializer=initializer,
                learn_prior=False)
        else:
            raise ValueError()

        tf.summary.scalar('mu_mean', tf.reduce_mean(mu))
        tf.summary.histogram('mu_posterior', mu)
        tf.summary.scalar('logsigmasq_mean', tf.reduce_mean(logsigmasq))
        tf.summary.histogram('logsigmasq_posterior', logsigmasq)
        if is_training:
            return draw_sample(mu=mu, logsigmasq=logsigmasq)
        else:
            return mu


def get_variable(
        name, shape, vparams: VariationalParams, dtype=tf.float32, initializer=None, is_training=True
):
    if vparams.enabled:
        return variational_variable(
            name=name,
            shape=shape,
            dtype=dtype,
            # initializer=tf.initializers.random_normal(stddev=0.1),
            initializer=initializer,
            is_training=is_training,
            vparams=vparams
        )
    else:
        return tf.get_variable(
            name=name,
            shape=shape if initializer is None or callable(initializer) else None,
            dtype=dtype,
            initializer=initializer
        )
