import tensorflow as tf

from .variational_variable import VariationalParams, variational_variable


def variational_fully_connected(
        inputs,
        num_outputs,
        vparams: VariationalParams,
        scope='fully_connected'):
    num_inputs = inputs.shape[-1].value
    with tf.variable_scope(scope):
        kernel = variational_variable(
            shape=(num_inputs, num_outputs),
            name='kernel',
            initializer=tf.initializers.glorot_uniform,
            vparams=vparams
        )
        bias = variational_variable(
            shape=(num_outputs,),
            name='bias',
            initializer=tf.initializers.zeros,
            vparams=vparams
        )
        output = tf.matmul(inputs, kernel) + bias
        assert inputs.shape.ndims == output.shape.ndims
        return output
