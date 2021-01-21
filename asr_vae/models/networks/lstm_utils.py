import tensorflow as tf
from tensorflow.contrib.cudnn_rnn.python.layers.cudnn_rnn import CudnnLSTM

from .basic_rnn import basic_lstm, basic_rnn
from .fully_connected import fully_connected_layer
from .variational.variational_lstm import CUDNN_RNN_BIDIRECTION, variational_lstm
from .variational.variational_variable import VariationalParams, get_variable


def ciel_div(x, y):
    return -tf.floor_div(-x, y)


def get_num_states(rnn_mode='lstm'):
    if rnn_mode == 'cudnn_lstm':
        return 2
    elif rnn_mode == 'cudnn_gru':
        return 1
    elif rnn_mode == 'cudnn_rnn_tanh':
        return 1
    elif rnn_mode == 'cudnn_rnn_relu':
        return 1
    else:
        raise ValueError()


def make_initial_states(
        batch_size, num_layers, dim, vparams: VariationalParams,
        bidirectional=False,
        is_training=True, rnn_mode='lstm'):
    num_dirs = 2 if bidirectional else 1
    num_states = get_num_states(rnn_mode=rnn_mode)
    states = []
    for i in range(num_states):
        initial_c = get_variable(
            name='initial_state_{}'.format(i),
            dtype=tf.float32,
            initializer=tf.initializers.zeros,
            shape=(num_layers * num_dirs, 1, dim),
            is_training=is_training,
            vparams=vparams)
        states.append(initial_c)

    states = [
        tf.tile(s, [1, batch_size, 1])
        for s in states
    ]
    while len(states) < 2:
        states.append([])

    return tuple(states)


def bilstm_residual_layer(
        x, num_units, vparams: VariationalParams, sequence_lengths=None, dropout=0., is_training=True,
        batch_norm_fn=None, rnn_mode='lstm'
):
    if x.shape[-1].value == num_units:
        skip = x
    else:
        skip = fully_connected_layer(
            activation_fn=None,
            inputs=x,
            num_outputs=num_units,
            vparams=vparams,
            scope='skip'
        )
    h = x
    if batch_norm_fn is not None:
        with tf.variable_scope('bn_1'):
            h = batch_norm_fn(h)
    h, states = bilstm_vanilla(
        x=h,
        num_layers=1,
        num_units=num_units,
        sequence_lengths=sequence_lengths,
        dropout=dropout,
        is_training=is_training,
        vparams=vparams,
        rnn_mode=rnn_mode
    )
    if batch_norm_fn is not None:
        with tf.variable_scope('bn_2'):
            h = batch_norm_fn(h)
    delta = fully_connected_layer(
        activation_fn=None,
        inputs=h,
        num_outputs=num_units,
        vparams=vparams,
        scope='delta'
    )
    h = delta + skip
    return h, states


def bilstm_stack(
        inputs, lstm_fn, num_layers, batch_norm_fn=None, dropout_fn=None
):
    h = inputs
    states = []
    with tf.variable_scope('bilstm_stack'):
        for i in range(num_layers):
            with tf.variable_scope('bilstm_layer_{}'.format(i)):
                h, state = lstm_fn(h)
                if batch_norm_fn is not None:
                    h = batch_norm_fn(h)
                if dropout_fn is not None:
                    h = dropout_fn(h)
            states.append(state)
    states = tf.concat(states, axis=-1)
    # if batch_norm_fn is not None:
    #    states = batch_norm_fn(states)
    # if dropout_fn is not None:
    #    states = dropout_fn(states)
    return h, states


def cudnn_lstm(inputs, num_layers, num_units, dropout, initial_states, is_training=True,
               direction=CUDNN_RNN_BIDIRECTION,
               sequence_lengths=None):
    lstm = CudnnLSTM(
        num_layers=num_layers,
        num_units=num_units,
        direction=direction,
        dropout=dropout
    )
    return lstm(
        inputs,
        sequence_lengths=sequence_lengths,
        initial_state=initial_states,
        training=is_training)


def bilstm_vanilla(x, num_layers, num_units, vparams: VariationalParams,
                   sequence_lengths=None, dropout=0., is_training=True,
                   direction=CUDNN_RNN_BIDIRECTION, rnn_mode="lstm"):
    if rnn_mode == 'tf_rnn_clipped_relu':
        return basic_rnn(
            inputs=x,
            num_units=num_units,
            vparams=vparams,
            sequence_length=sequence_lengths,
            direction=direction,
            num_layers=num_layers
        )
    elif rnn_mode == 'tf_lstm':
        return basic_lstm(
            inputs=x,
            num_units=num_units,
            vparams=vparams,
            sequence_length=sequence_lengths,
            direction=direction,
            num_layers=num_layers
        )

    assert x.shape.ndims == 3
    batch_size = tf.shape(x)[1]
    initial_states = make_initial_states(
        batch_size=batch_size,
        num_layers=num_layers,
        dim=num_units,
        bidirectional=True,
        vparams=vparams,
        is_training=is_training,
        rnn_mode=rnn_mode
    )

    y, yfinal = variational_lstm(
        inputs=x,
        num_layers=num_layers,
        num_units=num_units,
        scope='lstm',
        direction=direction,
        sequence_lengths=sequence_lengths,
        kernel_initializer=None,
        bias_initializer=None,
        initial_state=initial_states,
        dropout=dropout,
        is_training=is_training,
        vparams=vparams,
        rnn_mode=rnn_mode
    )
    """
    else:
        y, yfinal = cudnn_lstm(
            inputs=x,
            num_layers=num_layers,
            num_units=num_units,
            direction=direction,
            sequence_lengths=sequence_lengths,
            dropout=dropout,
            is_training=is_training,
            initial_states=initial_states
        )
    """
    print("yfinal1: {}".format(yfinal))
    yfinal = yfinal[0]
    # layers, n, dim
    print("Y: {}".format(y))
    print("yfinal: {}".format(yfinal))
    yfinal = tf.transpose(yfinal, (1, 0, 2))
    yfinal = tf.reshape(yfinal, (tf.shape(yfinal)[0], yfinal.shape[1].value * yfinal.shape[2].value))
    return y, yfinal


def sequence_pyramid(inputs, mode='concat'):
    assert inputs.shape.ndims == 3
    l = tf.shape(inputs)[0]
    n = tf.shape(inputs)[1]
    d = inputs.shape[2].value
    padding = tf.mod(l, 2)
    lpad = l + padding
    l2 = tf.floordiv(lpad, 2)
    outputs = tf.transpose(inputs, (1, 0, 2))
    outputs = tf.pad(outputs, [[0, 0], [0, padding], [0, 0]])
    if mode == 'concat':
        outputs = tf.reshape(outputs, [n, l2, 2 * d])
    elif mode == 'mean':
        outputs = tf.reshape(outputs, [n, l2, 2, d])
        outputs = tf.reduce_mean(outputs, axis=2)
    else:
        raise ValueError()
    outputs = tf.transpose(outputs, [1, 0, 2])
    return outputs


def sequence_pyramid_lengths(sequence_lengths):
    padding = tf.mod(sequence_lengths, 2)
    lpad = sequence_lengths + padding
    l2 = tf.floordiv(lpad, 2)
    return l2


RNN_MODES = ['cudnn_lstm', 'cudnn_gru', 'cudnn_rnn_tanh', 'cudnn_rnn_relu', 'tf_lstm', 'tf_rnn_clipped_relu']


def bilstm(
        inputs, num_layers, num_units,
        vparams: VariationalParams,
        sequence_lengths=None, is_training=True,
        residual=False, dropout_fn=None,
        batch_norm_fn=None, rnn_mode='invalid'):
    if rnn_mode not in RNN_MODES:
        raise ValueError("Unknown mode [{}] not in {}".format(rnn_mode, RNN_MODES))
    assert rnn_mode in RNN_MODES
    if (
            (not residual) and
            (batch_norm_fn is None) and
            (dropout_fn is None) and
            rnn_mode in ['cudnn_lstm', 'cudnn_gru', 'cudnn_rnn_tanh', 'cudnn_rnn_relu']
    ):
        return bilstm_vanilla(
            x=inputs,
            num_layers=num_layers,
            num_units=num_units,
            sequence_lengths=sequence_lengths,
            is_training=is_training,
            vparams=vparams,
            rnn_mode=rnn_mode
        )
    else:
        if residual:
            def lstm_fn(lstm_inputs):
                return bilstm_residual_layer(
                    x=lstm_inputs,
                    num_units=num_units,
                    sequence_lengths=sequence_lengths,
                    is_training=is_training,
                    vparams=vparams,
                    batch_norm_fn=batch_norm_fn,
                    rnn_mode=rnn_mode
                )
        else:
            def lstm_fn(lstm_inputs):
                return bilstm_vanilla(
                    x=lstm_inputs,
                    num_units=num_units,
                    sequence_lengths=sequence_lengths,
                    is_training=is_training,
                    vparams=vparams,
                    num_layers=1,
                    rnn_mode=rnn_mode
                )
        h, states = bilstm_stack(
            inputs=inputs,
            num_layers=num_layers,
            lstm_fn=lstm_fn,
            dropout_fn=dropout_fn,
            batch_norm_fn=None if residual else batch_norm_fn
        )
        if residual and batch_norm_fn:
            with tf.variable_scope('final_bn'):
                h = batch_norm_fn(h)
        return h, states


def sequence_index(lengths, maxlen):
    mask = tf.sequence_mask(
        lengths=lengths,
        maxlen=maxlen
    )
    mask_t = tf.transpose(mask, (1, 0))
    idx = tf.where_v2(mask_t)
    return idx
