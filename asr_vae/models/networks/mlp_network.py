import tensorflow as tf

from tensorflow.contrib import slim


def mlp_network(
        inputs,
        depth,
        hidden_dim,
        output_dim,
        hidden_activation_fn=tf.nn.leaky_relu,
        output_activation_fn=None):
    with tf.variable_scope('mlp'):
        h = inputs
        for i in range(depth):
            h=slim.fully_connected(
                inputs=h,
                num_outputs=hidden_dim,
                activation_fn=hidden_activation_fn,
                scope='mlp_{}'.format(i)
            )
        h = slim.fully_connected(
            inputs=h,
            num_outputs=output_dim,
            activation_fn=output_activation_fn,
            scope='mlp_output'
        )
    return h
