import tensorflow as tf
from tensorflow_gan.python.train import RunTrainOpsHook
from .make_train_op import get_total_loss, make_train_op
from ..anneal import get_scale


def gan_losses(yreal, yfake):
    dis_real = tf.reduce_mean(tf.nn.softplus(-yreal))
    dis_fake = tf.reduce_mean(tf.nn.softplus(yfake))
    gen_fake = tf.reduce_mean(tf.nn.softplus(-yfake))
    dis_loss = (dis_real + dis_fake) * 0.5
    gen_loss = gen_fake
    return dis_loss, gen_loss


def gan_mod_losses(yreal, yfake):
    """
    dis_real = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.ones(shape=tf.shape(yreal), dtype=tf.int32),
        logits=yreal
    )
    dis_fake = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.zeros(shape=tf.shape(yfake), dtype=tf.int32),
        logits=yfake
    )
    gen_fake = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.ones(shape=tf.shape(yfake), dtype=tf.int32),
        logits=yfake
    )
    sig(x) = 1/(1+e^{-x})
    1-sig(x) = e^{-x}/(1+e^{-x}) = 1/(e^x+1)
    log(sig(x)) = -log(1+e^{-x})
    log(1-sig(x)) = -log(1+e^x)
    """
    dis_real = tf.reduce_mean(tf.nn.softplus(-yreal))
    dis_fake = tf.reduce_mean(tf.nn.softplus(yfake))
    # gen_fake = tf.reduce_mean(tf.nn.softplus(-yfake))
    gen_fake = tf.reduce_mean(-tf.nn.softplus(yfake))
    dis_loss = (dis_real + dis_fake) * 0.5
    gen_loss = gen_fake
    return dis_loss, gen_loss


def wgan_losses(yreal, yfake):
    mean_yreal = tf.reduce_mean(yreal)
    mean_yfake = tf.reduce_mean(yfake)
    dis_loss = mean_yreal - mean_yfake
    gen_loss = mean_yfake
    return dis_loss, gen_loss


def aae_losses(yreal, yfake, params):
    if params.aae_mode == 'wgan':
        dis_loss, gen_loss = wgan_losses(yreal=yreal, yfake=yfake)
    elif params.aae_mode == 'gan':
        dis_loss, gen_loss = gan_losses(yreal=yreal, yfake=yfake)
    elif params.aae_mode == 'gan_mode':
        dis_loss, gen_loss = gan_mod_losses(yreal=yreal, yfake=yfake)
    else:
        raise NotImplementedError()
    scale = get_scale(params=params)
    gen_loss = gen_loss * scale
    return dis_loss, gen_loss


def aae_losses_and_hooks(real, fake, discriminator_fn, model_scope, params, axis=0):
    n = tf.shape(real)[axis]
    ndim = real.shape.ndims
    with tf.variable_scope('discriminator_scope', reuse=False) as discriminator_scope:
        with tf.name_scope('discriminator_scope/real/'):
            yreal = discriminator_fn(real)
    with tf.variable_scope(discriminator_scope, reuse=True):
        with tf.name_scope('discriminator_scope/fake/'):
            yfake = discriminator_fn(fake)
    with tf.variable_scope(discriminator_scope, reuse=True):
        with tf.name_scope('discriminator_scope/inter/'):
            shape = ([1] * axis) + [n] + ([1] * (ndim - axis - 1))
            alpha = tf.random.uniform(shape=shape, dtype=tf.float32)
            print("Alpha: {}".format(alpha))
            inter = ((1. - alpha) * real) + (alpha * fake)
            yinter = discriminator_fn(inter)

    with tf.name_scope('aae_losses'):
        inter_grad = tf.gradients(
            tf.reduce_sum(yinter), inter
        )
        axes = list(range(0, ndim))
        axes.remove(axis)
        # inter_grad_norm = tf.norm(inter_grad, ord=2, axis=axes)
        inter_grad_norm = tf.sqrt(
            tf.reduce_sum(
                tf.square(inter_grad),
                axis=axes)
        )
        grad_penalty = params.penalty_weight * tf.reduce_mean(tf.squared_difference(inter_grad_norm, 1.))

        dis_loss, gen_loss = aae_losses(
            yreal=yreal,
            yfake=yfake,
            params=params
        )

    with tf.name_scope(model_scope + '/'):
        gen_loss = tf.identity(gen_loss)
        tf.summary.scalar('gen_loss', gen_loss)
        tf.losses.add_loss(gen_loss)
    with tf.name_scope(discriminator_scope.name + '/'):
        dis_loss = tf.identity(dis_loss)
        tf.summary.scalar('dis_loss', dis_loss)
        tf.losses.add_loss(dis_loss)
        grad_penalty = tf.identity(grad_penalty)
        tf.summary.scalar('grad_penalty', grad_penalty)
        tf.losses.add_loss(grad_penalty)
        dis_total_loss = get_total_loss(scope=discriminator_scope.name)
        tf.summary.scalar('dis_total_loss', dis_total_loss)
    dis_train_op = make_train_op(
        scope=discriminator_scope.name,
        lr=params.dis_lr,
        params=params,
        global_step=None,
        total_loss=dis_total_loss,
        opt=params.optimizer
    )
    dis_train_hook = RunTrainOpsHook(
        train_ops=[dis_train_op],
        train_steps=params.discriminator_steps
    )
    return dis_train_hook
