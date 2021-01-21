import tensorflow as tf
from tensorflow.contrib import slim


def resblock_2d(inputs, kernel_size, num_outputs, stride, scope, activation_fn=tf.nn.leaky_relu):
    num_inputs = inputs.shape[-1].value
    assert num_inputs, "known final dim required"
    with tf.variable_scope(scope):
        h1 = slim.conv2d(
            inputs=inputs,
            activation_fn=activation_fn,
            kernel_size=kernel_size,
            stride=stride,
            num_outputs=num_outputs,
            scope='block1',
            padding='SAME'
        )
        h2 = slim.conv2d(
            inputs=h1,
            activation_fn=None,
            kernel_size=kernel_size,
            stride=(1, 1),
            num_outputs=num_outputs,
            scope='block2',
            padding='SAME'
        )
        if num_inputs == num_outputs:
            ident = inputs
        else:
            ident = slim.conv2d(
                inputs=inputs,
                activation_fn=None,
                kernel_size=1,
                stride=stride,
                num_outputs=num_outputs,
                scope='ident_connection',
                padding='SAME'
            )

        h3 = ident + h2
        y = activation_fn(h3)
    return y


def resblock_1d(inputs, kernel_size, num_outputs, stride, scope, activation_fn=tf.nn.leaky_relu, is_training=True):
    num_inputs = inputs.shape[-1].value
    assert num_inputs, "known final dim required"
    with tf.variable_scope(scope):
        inputs = slim.batch_norm(inputs=inputs, is_training=is_training, scope='input_bn')
        h1 = slim.conv1d(
            inputs=inputs,
            activation_fn=activation_fn,
            kernel_size=kernel_size,
            stride=stride,
            num_outputs=num_outputs,
            scope='block1',
            padding='SAME'
        )
        h1 = slim.batch_norm(inputs=h1, is_training=is_training, scope='h1_bn')
        h2 = slim.conv1d(
            inputs=h1,
            activation_fn=None,
            kernel_size=kernel_size,
            stride=1,
            num_outputs=num_outputs,
            scope='block2',
            padding='SAME'
        )
        if num_inputs == num_outputs:
            ident = inputs
        else:
            ident = slim.conv1d(
                inputs=inputs,
                activation_fn=None,
                kernel_size=1,
                stride=stride,
                num_outputs=num_outputs,
                scope='ident_connection',
                padding='SAME'
            )

        h3 = ident + h2
        y = activation_fn(h3)
    return y
