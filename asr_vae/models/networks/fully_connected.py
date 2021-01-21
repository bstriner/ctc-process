import tensorflow as tf
from tensorflow.contrib import slim

from .variational.variational_fully_connected import variational_fully_connected
from .variational.variational_variable import VariationalParams


def fully_connected_layer(
        inputs,
        num_outputs,
        vparams: VariationalParams,
        activation_fn=tf.nn.leaky_relu,
        batch_norm_fn=None,
        scope="fully_connected"
):
    with tf.variable_scope(scope):
        if vparams.enabled:
            h = variational_fully_connected(
                inputs=inputs,
                num_outputs=num_outputs,
                scope='fully_connected',
                vparams=vparams
            )
        else:
            h = slim.fully_connected(
                inputs=inputs,
                num_outputs=num_outputs,
                activation_fn=None,
                scope='fully_connected'
            )
        if batch_norm_fn is not None:
            h = batch_norm_fn(h)
        if activation_fn is not None:
            h = activation_fn(h)
        return h


def fully_connected_stack(
        inputs,
        num_outputs,
        activation_fn=tf.nn.leaky_relu,
        batch_norm_fn=None,
        residual=False,
        variational=False,
        variational_sigma=0.075,
        scope="fully_connected_stack",
        num_layers=1
):
    h = inputs
    with tf.variable_scope(scope):
        for i in range(num_layers):
            h = fully_connected(
                inputs=inputs,
                num_outputs=num_outputs,
                activation_fn=activation_fn,
                batch_norm_fn=batch_norm_fn,
                variational=variational,
                variational_sigma=variational_sigma,
                residual=residual,
                scope="fully_connected_layer_{}".format(i)
            )
    return h


def fully_connected(
        inputs,
        num_outputs,
        activation_fn=tf.nn.leaky_relu,
        batch_norm_fn=None,
        residual=False,
        vparams=None,
        scope="fully_connected"
):
    if residual:
        return fully_connected_residual(
            inputs=inputs,
            num_outputs=num_outputs,
            activation_fn=activation_fn,
            batch_norm_fn=batch_norm_fn,
            scope=scope,
            vparams=vparams
        )
    else:
        return fully_connected_layer(
            inputs=inputs,
            num_outputs=num_outputs,
            activation_fn=activation_fn,
            batch_norm_fn=batch_norm_fn,
            scope=scope,
            vparams=vparams
        )


def fully_connected_residual(
        inputs,
        num_outputs,
        activation_fn=tf.nn.leaky_relu,
        batch_norm_fn=None,
        scope="fully_connected",
        depth=1,
        vparams=None
):
    with tf.variable_scope(scope):
        if inputs.shape[-1].value == num_outputs:
            skip = inputs
        else:
            skip = fully_connected_layer(
                inputs=inputs,
                num_outputs=num_outputs,
                activation_fn=None,
                batch_norm_fn=None,
                scope='projection',
                vparams=vparams
            )
        h = inputs
        for i in range(depth):
            h = fully_connected_layer(
                inputs=h,
                num_outputs=num_outputs,
                activation_fn=activation_fn,
                batch_norm_fn=batch_norm_fn,
                scope='res_{}'.format(i),
                vparams=vparams
            )
        delta = fully_connected_layer(
            inputs=h,
            num_outputs=num_outputs,
            activation_fn=None,
            vparams=vparams,
            scope='delta'
        )
        outputs = skip + delta
        return outputs
