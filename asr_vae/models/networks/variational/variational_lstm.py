import numpy as np
import tensorflow as tf
from tensorflow.contrib.cudnn_rnn.python.layers.cudnn_rnn import CUDNN_GRU, CUDNN_GRU_PARAMS_PER_LAYER, \
    CUDNN_INPUT_LINEAR_MODE, CUDNN_LSTM, CUDNN_LSTM_PARAMS_PER_LAYER, CUDNN_RNN_BIDIRECTION, CUDNN_RNN_RELU, \
    CUDNN_RNN_RELU_PARAMS_PER_LAYER, CUDNN_RNN_TANH, CUDNN_RNN_TANH_PARAMS_PER_LAYER, CUDNN_RNN_UNIDIRECTION
from tensorflow.contrib.cudnn_rnn.python.ops.cudnn_rnn_ops import _cudnn_rnn, cudnn_rnn_canonical_to_opaque_params

from .variational_variable import VariationalParams, get_variable


def state_shape(num_layers, num_dirs, num_units, batch_size):
    """Shape of Cudnn LSTM states.
    Shape is a 2-element tuple. Each is
    [num_layers * num_dirs, batch_size, num_units]
    Args:
      batch_size: an int
    Returns:
      a tuple of python arrays.
    """
    return ([num_layers * num_dirs, batch_size, num_units],
            [num_layers * num_dirs, batch_size, num_units])


def get_num_params_per_layer(rnn_mode=CUDNN_LSTM):
    if rnn_mode == CUDNN_LSTM:
        return CUDNN_LSTM_PARAMS_PER_LAYER
    elif rnn_mode == CUDNN_GRU:
        return CUDNN_GRU_PARAMS_PER_LAYER
    elif rnn_mode == CUDNN_RNN_TANH:
        return CUDNN_RNN_TANH_PARAMS_PER_LAYER
    elif rnn_mode == CUDNN_RNN_RELU:
        return CUDNN_RNN_RELU_PARAMS_PER_LAYER
    else:
        raise ValueError()


def canonical_bias_shape_fn(direction, num_outputs, rnn_mode=CUDNN_LSTM):
    num_dirs = 1 if direction == CUDNN_RNN_UNIDIRECTION else 2
    num_params_per_layer = get_num_params_per_layer(rnn_mode=rnn_mode)
    return [[num_outputs]] * num_dirs * num_params_per_layer


def canonical_bias_shapes_fn(direction, num_outputs, layers, rnn_mode=CUDNN_LSTM):
    return canonical_bias_shape_fn(direction=direction, num_outputs=num_outputs, rnn_mode=rnn_mode) * layers


def canonical_weight_shape_fn(input_size, direction, layer, num_outputs, rnn_mode=CUDNN_LSTM):
    num_params_per_layer = get_num_params_per_layer(rnn_mode=rnn_mode)
    num_gates = num_params_per_layer // 2
    is_bidi = direction == CUDNN_RNN_BIDIRECTION
    if layer == 0:
        wts_applied_on_inputs = [(num_outputs, input_size)] * num_gates
    else:
        if is_bidi:
            wts_applied_on_inputs = [(num_outputs, 2 * num_outputs)] * num_gates
        else:
            wts_applied_on_inputs = [(num_outputs, num_outputs)] * num_gates
    wts_applied_on_hidden_states = [(num_outputs, num_outputs)] * num_gates
    tf_wts = wts_applied_on_inputs + wts_applied_on_hidden_states
    return tf_wts if not is_bidi else tf_wts * 2


def canonical_weight_shapes_fn(input_size, direction, layers, num_outputs, rnn_mode=CUDNN_LSTM):
    shapes = []
    for layer in range(layers):
        shapes.extend(canonical_weight_shape_fn(input_size, direction, layer, num_outputs, rnn_mode=rnn_mode))
    return shapes


def zero_state(num_layers, num_dirs, num_units, batch_size, dtype):
    res = []
    for sp in state_shape(
            num_layers=num_layers,
            num_dirs=num_dirs,
            num_units=num_units,
            batch_size=batch_size):
        res.append(tf.zeros(sp, dtype=dtype))
    return tuple(res)


def variational_lstm(inputs, num_units, scope, vparams: VariationalParams, num_layers=1,
                     direction=CUDNN_RNN_BIDIRECTION, sequence_lengths=None, rnn_mode="lstm",
                     kernel_initializer=None, bias_initializer=None, initial_state=None, dropout=0.0, is_training=True):
    if rnn_mode == 'cudnn_lstm':
        rnn_mode = CUDNN_LSTM
    elif rnn_mode == 'cudnn_gru':
        rnn_mode = CUDNN_GRU
    elif rnn_mode == 'cudnn_rnn_tanh':
        rnn_mode = CUDNN_RNN_TANH
    elif rnn_mode == 'cudnn_rnn_relu':
        rnn_mode = CUDNN_RNN_RELU
    else:
        raise ValueError()
    with tf.variable_scope(scope):
        dtype = tf.float32
        batch_size = tf.shape(inputs)[1]
        input_shape = inputs.shape
        if input_shape.ndims != 3:
            raise ValueError("Expecting input_shape with 3 dims, got %d" %
                             input_shape.ndims)
        if input_shape[-1].value is None:
            raise ValueError("The last dimension of the inputs to `CudnnRNN` "
                             "should be defined. Found `None`.")
        input_size = input_shape[-1].value

        if kernel_initializer is None:
            kernel_initializer = tf.initializers.glorot_uniform(dtype=tf.float32)

        if bias_initializer is None:
            bias_initializer = tf.initializers.constant(0.0, dtype=tf.float32)

        canonical_weight_shapes = canonical_weight_shapes_fn(
            input_size=input_size,
            direction=direction,
            layers=num_layers,
            num_outputs=num_units,
            rnn_mode=rnn_mode)
        canonical_bias_shapes = canonical_bias_shapes_fn(
            direction=direction,
            num_outputs=num_units,
            layers=num_layers,
            rnn_mode=rnn_mode)
        weights = [
            kernel_initializer(sp, dtype=dtype)
            for sp in canonical_weight_shapes
        ]
        biases = [
            bias_initializer(sp, dtype=dtype)
            for sp in canonical_bias_shapes
        ]
        print(canonical_weight_shapes)
        print(canonical_bias_shapes)
        print(len(canonical_weight_shapes))
        print(len(canonical_bias_shapes))
        opaque_params_t = cudnn_rnn_canonical_to_opaque_params(
            rnn_mode=rnn_mode,
            num_layers=num_layers,
            num_units=num_units,
            input_size=input_size,
            weights=weights,
            biases=biases,
            input_mode=CUDNN_INPUT_LINEAR_MODE,
            direction=direction,
            dropout=0,
            seed=0,
            name=None)
        count = 0
        for weight in weights:
            for s in weight.shape:
                assert s
            count += np.product([s.value for s in weight.shape])
        for bias in biases:
            for s in bias.shape:
                assert s
            count += np.product([s.value for s in bias.shape])
        print("Count: {}".format(count))
        tfcount = tf.constant([count])
        with tf.control_dependencies([tf.assert_equal(tf.shape(opaque_params_t), tfcount)]):
            opaque_params_t = tf.identity(opaque_params_t)

        print("opaque_params_t: {}".format(opaque_params_t))
        opaque_params_t.set_shape([count])
        print("opaque_params_t: {}".format(opaque_params_t))

        opaque_params = get_variable(
            shape=opaque_params_t.shape,
            name='kernel',
            initializer=opaque_params_t,
            vparams=vparams
        )
        print("LSTM Kernel: {}".format(opaque_params))

        if initial_state is not None and not isinstance(initial_state, tuple):
            raise TypeError("Invalid initial_state type: %s, expecting tuple." %
                            initial_state)
        num_dirs = 2 if direction == CUDNN_RNN_BIDIRECTION else 1
        if initial_state is None:
            initial_state = zero_state(
                num_layers=num_layers,
                num_dirs=num_dirs,
                num_units=num_units,
                batch_size=batch_size,
                dtype=dtype)
        h, c = initial_state  # pylint:disable=unbalanced-tuple-unpacking,unpacking-non-sequence

        output, output_h, output_c = _cudnn_rnn(  # pylint:disable=protected-access
            inputs=inputs,
            input_h=h,
            input_c=c,
            rnn_mode=rnn_mode,
            params=opaque_params,
            is_training=is_training,
            sequence_lengths=sequence_lengths,
            time_major=True,
            input_mode=CUDNN_INPUT_LINEAR_MODE,
            direction=direction,
            dropout=dropout)
        return output, (output_h, output_c)
