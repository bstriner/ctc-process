import tensorflow as tf
from tensorflow.contrib import slim


def residual_fn_v1(inputs, kernel_size, dilation_rate=1, is_training=True, activation_fn=tf.nn.leaky_relu):
    num_outputs = inputs.shape[-1].value
    assert num_outputs
    res = inputs
    res = slim.batch_norm(res, scope='bn_1', is_training=is_training)
    res = activation_fn(res)
    res = slim.conv1d(
        inputs=res,
        scope='weight_1',
        num_outputs=num_outputs,
        activation_fn=None,
        kernel_size=kernel_size,
        rate=dilation_rate
    )
    res = slim.batch_norm(res, scope='bn_2', is_training=is_training)
    res = activation_fn(res)
    res = slim.fully_connected(
        inputs=res,
        scope='weight_2',
        num_outputs=num_outputs,
        activation_fn=None
    )
    return res


def residual_fn_v2(inputs, kernel_size, dilation_rate=1, is_training=True, activation_fn=tf.nn.leaky_relu):
    num_outputs = inputs.shape[-1].value
    assert num_outputs
    res = inputs
    res = slim.conv1d(
        inputs=res,
        scope='weight_1',
        num_outputs=num_outputs,
        activation_fn=None,
        kernel_size=kernel_size,
        rate=dilation_rate
    )
    res = slim.batch_norm(res, scope='bn_2', is_training=is_training)
    res = activation_fn(res)
    res = slim.fully_connected(
        inputs=res,
        scope='weight_2',
        num_outputs=num_outputs,
        activation_fn=None
    )
    return res


def resnet_block_v1(inputs, scope, kernel_size, is_training=True, activation_fn=tf.nn.leaky_relu):
    with tf.variable_scope(scope):
        res = residual_fn_v1(
            inputs=inputs,
            kernel_size=kernel_size,
            is_training=is_training,
            activation_fn=activation_fn
        )
    return inputs + res


def resnet_block_v2(inputs, scope, kernel_size, is_training=True, activation_fn=tf.nn.leaky_relu):
    with tf.variable_scope(scope):
        res = inputs
        res = slim.batch_norm(res, scope='bn_input', is_training=is_training)
        res = activation_fn(res)
        with tf.variable_scope('dilation_1'):
            res1 = residual_fn_v2(
                inputs=res,
                kernel_size=kernel_size,
                is_training=is_training,
                activation_fn=activation_fn
            )
        with tf.variable_scope('dilation_3'):
            res3 = residual_fn_v2(
                inputs=res,
                kernel_size=kernel_size,
                is_training=is_training,
                activation_fn=activation_fn,
                dilation_rate=3
            )
    return inputs + res1 + res3


def resnet_block_v3(inputs, scope, kernel_size, is_training=True, activation_fn=tf.nn.leaky_relu):
    with tf.variable_scope(scope):
        res = inputs
        res = slim.batch_norm(res, scope='bn_input', is_training=is_training)
        res = activation_fn(res)
        with tf.variable_scope('dilation_1'):
            res1 = residual_fn_v2(
                inputs=res,
                kernel_size=kernel_size,
                is_training=is_training,
                activation_fn=activation_fn
            )
        with tf.variable_scope('dilation_3'):
            res3 = residual_fn_v2(
                inputs=res,
                kernel_size=kernel_size,
                is_training=is_training,
                activation_fn=activation_fn,
                dilation_rate=3
            )
        with tf.variable_scope('dilation_5'):
            res5 = residual_fn_v2(
                inputs=res,
                kernel_size=kernel_size,
                is_training=is_training,
                activation_fn=activation_fn,
                dilation_rate=5
            )
    return inputs + res1 + res3 + res5
