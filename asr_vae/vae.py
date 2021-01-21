import tensorflow as tf


def vae_sample(mu, logsigmasq, utterance_mask=None):
    """

    :param mu: (L,N, D)
    :param logsigmasq:
    :return:
    """

    sigmasq = tf.exp(logsigmasq, name='sigmasq')
    sigma = tf.sqrt(sigmasq, name='sigma')
    z = mu + (sigma * tf.random_normal(
        shape=tf.shape(mu),
        mean=0,
        stddev=1,
        dtype=tf.float32))
    kl = 0.5 * tf.square(mu) + sigmasq - logsigmasq - 1
    if utterance_mask is not None:
        kl = kl * tf.cast(tf.expand_dims(utterance_mask, 2), tf.float32)
    kl = tf.reduce_sum(kl)
    tf.losses.add_loss(kl, loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)
    tf.summary.scalar("KL_Loss", kl)
    tf.add_to_collection('KL', kl)
    return z
