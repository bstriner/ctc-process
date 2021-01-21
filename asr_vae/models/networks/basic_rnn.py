import tensorflow as tf

from .variational.variational_lstm import CUDNN_RNN_BIDIRECTION
from .variational.variational_variable import VariationalParams, get_variable


def clipped_relu(input):
    return tf.clip_by_value(input, 0, 20)


def rnn_initial_states(batch_size, state_size, vparams: VariationalParams, name='state', is_training=True):
    assert isinstance(state_size, int)

    initial_c = get_variable(
        name=name,
        dtype=tf.float32,
        initializer=tf.initializers.zeros,
        shape=(1, state_size),
        is_training=is_training,
        vparams=vparams)
    initial_c = tf.tile(initial_c, [batch_size, 1])
    return initial_c


def lstm_initial_states(batch_size, state_size, vparams: VariationalParams, name='state', is_training=True):
    assert isinstance(state_size, tuple)
    assert len(state_size) == 2
    c = rnn_initial_states(
        batch_size=batch_size,
        state_size=state_size[0],
        vparams=vparams,
        name="{}_c".format(name),
        is_training=is_training
    )
    h = rnn_initial_states(
        batch_size=batch_size,
        state_size=state_size[1],
        vparams=vparams,
        name="{}_h".format(name),
        is_training=is_training
    )
    return tf.contrib.rnn.LSTMStateTuple(h=h, c=c)


def basic_rnn(inputs, num_units, num_layers, vparams: VariationalParams, sequence_length=None,
              direction=CUDNN_RNN_BIDIRECTION):
    assert num_layers == 1
    assert direction == CUDNN_RNN_BIDIRECTION
    assert not vparams.enabled
    cell_fw = tf.contrib.rnn.BasicRNNCell(
        num_units=num_units,
        activation=clipped_relu,
        name='cell_fw'
    )
    cell_bw = tf.contrib.rnn.BasicRNNCell(
        num_units=num_units,
        activation=clipped_relu,
        name='cell_bw'
    )
    batch_size = tf.shape(inputs)[1]
    initial_state_fw = rnn_initial_states(
        batch_size=batch_size,
        state_size=cell_fw.state_size,
        name='initial_state_fw',
        vparams=vparams)
    initial_state_bw = rnn_initial_states(
        batch_size=batch_size,
        state_size=cell_bw.state_size,
        name='initial_state_bw',
        vparams=vparams)
    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=cell_fw,
        cell_bw=cell_bw,
        inputs=inputs,
        sequence_length=sequence_length,
        initial_state_fw=initial_state_fw,
        initial_state_bw=initial_state_bw,
        time_major=True
    )
    outputs = tf.concat(outputs, axis=-1)
    return outputs, output_states


def basic_lstm(inputs, num_units, num_layers, vparams: VariationalParams, sequence_length=None,
               direction=CUDNN_RNN_BIDIRECTION):
    assert num_layers == 1
    assert direction == CUDNN_RNN_BIDIRECTION
    assert not vparams.enabled
    cell_fw = tf.contrib.rnn.BasicLSTMCell(
        num_units=num_units,
        name='cell_fw'
    )
    cell_bw = tf.contrib.rnn.BasicLSTMCell(
        num_units=num_units,
        activation=clipped_relu,
        name='cell_bw'
    )
    batch_size = tf.shape(inputs)[1]
    initial_state_fw = lstm_initial_states(
        batch_size=batch_size,
        state_size=cell_fw.state_size,
        name='initial_state_fw',
        vparams=vparams)
    initial_state_bw = lstm_initial_states(
        batch_size=batch_size,
        state_size=cell_bw.state_size,
        name='initial_state_bw',
        vparams=vparams)
    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=cell_fw,
        cell_bw=cell_bw,
        inputs=inputs,
        sequence_length=sequence_length,
        initial_state_fw=initial_state_fw,
        initial_state_bw=initial_state_bw,
        time_major=True
    )
    outputs = tf.concat(outputs, axis=-1)
    output_states = output_states[0]
    return outputs, output_states
