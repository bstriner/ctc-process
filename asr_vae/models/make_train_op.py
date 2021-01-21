import tensorflow as tf
from tensorflow.contrib.training.python.training.training import create_train_op

from .helpers.transform_grads import make_transform_grads_fn


def get_total_loss(scope):
    with tf.name_scope(scope + "/"):
        losses = tf.losses.get_losses(scope=scope)
        print("{} losses: {}".format(scope, losses))
        losses += tf.losses.get_regularization_losses(scope=scope)
        total_loss = tf.add_n(losses)
        return total_loss


def make_opt(lr, params, opt):
    if opt == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    elif opt == 'adamw':
        optimizer = tf.contrib.opt.AdamWOptimizer(
            learning_rate=lr,
            weight_decay=params.l2)
    elif opt == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate=lr,
            rho=0.95,
            epsilon=1e-08)
    elif opt == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=lr,
            momentum=params.momentum,
            use_nesterov=False
        )
    elif opt == 'nesterov':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=lr,
            momentum=params.momentum,
            use_nesterov=True
        )
    elif opt == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=lr
        )
    else:
        raise ValueError()
    return optimizer


def make_train_op(scope, lr, params, global_step, total_loss, opt='adam'):
    optimizer = make_opt(lr=lr, params=params, opt=opt)
    transform_grads_fn = make_transform_grads_fn(params=params)
    variables = tf.trainable_variables(scope=scope)
    updates = tf.get_collection(key=tf.GraphKeys.UPDATE_OPS, scope=scope)
    train_op = create_train_op(
        total_loss=total_loss,
        optimizer=optimizer,
        update_ops=updates,
        variables_to_train=variables,
        transform_grads_fn=transform_grads_fn,
        # summarize_gradients=False,
        # aggregation_method=None,
        # check_numerics=True,
        global_step=global_step)
    return train_op
