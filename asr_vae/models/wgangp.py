import tensorflow as tf
from tensorflow.contrib import slim


def wgangp(
        xfake_packed,
        xreal_packed,
        pack_idx,
        n,
        length,
        latent_dim,
        discriminator_fn,
        dis_scope,
        aae_scope,
        params
):
    with tf.variable_scope(dis_scope, reuse=False):
        with tf.name_scope("Fake"):
            yfake = discriminator_fn(xfake_packed)
            yfake = tf.reduce_mean(yfake)
    with tf.variable_scope(dis_scope, reuse=True):
        with tf.name_scope("Real"):
            yreal = discriminator_fn(xreal_packed)
            yreal = tf.reduce_mean(yreal)

    with tf.name_scope("GP"):
        xreal_padded = tf.scatter_nd(
            updates=xreal_packed,
            indices=pack_idx,
            shape=(length, n, latent_dim)
        )
        xfake_padded = tf.scatter_nd(
            updates=xfake_packed,
            indices=pack_idx,
            shape=(length, n, latent_dim)
        )
        alpha = tf.random_uniform(shape=(1, n, 1))
        xint_padded = (alpha * xfake_padded) + ((1 - alpha) * xreal_padded)
        xint_padded = tf.stop_gradient(xint_padded)
        xint_packed = tf.gather_nd(
            params=xint_padded,
            indices=pack_idx
        )
    with tf.variable_scope(dis_scope, reuse=True):
        with tf.name_scope("Interp"):
            yint = discriminator_fn(xint_packed)
    with tf.name_scope("GP"):
        gint = tf.gradients(ys=tf.reduce_sum(yint), xs=[xint_padded])[0]
        g2int = tf.reduce_sum(tf.square(gint), axis=(0, 2))
        penalty = tf.reduce_mean(tf.square(1 - g2int))
        tf.losses.add_loss(penalty, loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)

    main_loss = tf.add_n(tf.losses.get_losses(scope=aae_scope.name, loss_collection=tf.GraphKeys.LOSSES))
    wgan_loss = yreal - yfake
    generator_loss = main_loss + wgan_loss
    discriminator_loss = penalty - wgan_loss

    gen_params = tf.trainable_variables(scope=aae_scope.name)
    gen_optimizer = tf.train.AdamOptimizer(params.gen_lr)
    gen_op = slim.learning.create_train_op(
        total_loss=generator_loss,
        variables_to_train=gen_params,
        optimizer=gen_optimizer
    )
    dis_params = tf.trainable_variables(scope=dis_scope.name)
    dis_optimizer = tf.train.AdamOptimizer(params.dis_lr)
    dis_op = slim.learning.create_train_op(
        total_loss=discriminator_loss,
        variables_to_train=dis_params,
        optimizer=dis_optimizer,
        global_step=None
    )

    tf.summary.scalar("wgan_loss", wgan_loss)
    tf.summary.scalar("wgan_penalty", penalty)
    tf.summary.scalar("generator_total_loss", generator_loss)
    tf.summary.scalar("discriminator_total_loss", discriminator_loss)

    return gen_op, dis_op
