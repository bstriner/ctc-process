import tensorflow as tf
from tensorflow.contrib import slim

from ...sparse import sparse_tensor_to_dense_scatter


def attention_fn_softmax_loop(
        utterance_h, utterance_lengths, transcript_h, transcript_lengths,
        dim, batch_first=False, weight_regularizer=None):
    if batch_first:
        ul = tf.shape(utterance_h)[1]
        tl = tf.shape(transcript_h)[1]
        n = tf.shape(utterance_h)[0]
    else:
        ul = tf.shape(utterance_h)[0]
        tl = tf.shape(transcript_h)[0]
        n = tf.shape(utterance_h)[1]
        utterance_h = tf.transpose(utterance_h, (1, 0, 2))
        transcript_h = tf.transpose(transcript_h, (1, 0, 2))

    utterance_key = slim.fully_connected(
        utterance_h,
        num_outputs=dim,
        activation_fn=None,
        weights_regularizer=weight_regularizer,
        scope='utterance_attn')  # (n, ul, d)

    transcript_key = slim.fully_connected(
        transcript_h,
        num_outputs=dim,
        activation_fn=None,
        weights_regularizer=weight_regularizer,
        scope='transcript_attn')  # (n, tl, d)
    #transcript_key = transcript_key / tf.norm(transcript_key, ord=2, axis=-1, keepdims=True)

    utterance_ta = tf.TensorArray(
        dtype=utterance_key.dtype,
        size=n
    ).unstack(utterance_key)
    utterance_lengths_ta = tf.TensorArray(
        dtype=utterance_lengths.dtype,
        size=n
    ).unstack(utterance_lengths)
    transcript_ta = tf.TensorArray(
        dtype=transcript_key.dtype,
        size=n
    ).unstack(transcript_key)
    transcript_lengths_ta = tf.TensorArray(
        dtype=transcript_lengths.dtype,
        size=n
    ).unstack(transcript_lengths)

    def cond(i, arr):
        return i < n

    def body(i, arr: tf.TensorArray):
        u_t = utterance_ta.read(i)
        ul_t = utterance_lengths_ta.read(i)
        t_t = transcript_ta.read(i)
        tl_t = transcript_lengths_ta.read(i)
        uc_t = u_t[:ul_t, :]
        tc_t = t_t[:tl_t, :]
        energy = tf.matmul(uc_t, tc_t, transpose_b=True)
        attn = tf.nn.softmax(energy, axis=-1)
        pattn = tf.pad(attn, [[0, ul - ul_t], [0, tl - tl_t]])
        return i + 1, arr.write(i, pattn)

    loop_vars_in = tf.constant(0), tf.TensorArray(size=n, dtype=utterance_h.dtype)
    loop_vars_out = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=loop_vars_in
    )
    return loop_vars_out[1].stack()


def attention_fn_softmax_loop2(utterance_h, utterance_lengths, transcript_h, transcript_lengths, dim,
                               batch_first=False, weight_regularizer=None):
    utterance_key = slim.fully_connected(
        utterance_h,
        num_outputs=dim,
        activation_fn=None,
        weights_regularizer=weight_regularizer,
        scope='utterance_attn')  # (ul, n, d))
    transcript_key = slim.fully_connected(
        transcript_h,
        num_outputs=dim,
        activation_fn=None,
        weights_regularizer=weight_regularizer,
        scope='transcript_attn')  # (tl, n, d)
    transcript_key = transcript_key / tf.norm(transcript_key, ord=2, axis=-1, keepdims=True)

    if batch_first:
        ul = tf.shape(utterance_h)[1]
        tl = tf.shape(transcript_h)[1]
        n = tf.shape(utterance_h)[0]
        energy = tf.matmul(
            utterance_key,  # (n, ul, d)
            tf.transpose(transcript_key, (0, 2, 1))  # (n, d, tl)
        )  # (n, ul, tl)
    else:
        ul = tf.shape(utterance_h)[0]
        tl = tf.shape(transcript_h)[0]
        n = tf.shape(utterance_h)[1]
        energy = tf.matmul(
            tf.transpose(utterance_key, (1, 0, 2)),  # (n, ul, d)
            tf.transpose(transcript_key, (1, 2, 0))  # (n, d, tl)
        )  # (n, ul, tl)

    def cond(i, arr):
        return i < n

    utterance_lengths_ta = tf.TensorArray(
        dtype=utterance_lengths.dtype,
        size=n).unstack(utterance_lengths)
    transcript_lengths_ta = tf.TensorArray(
        dtype=transcript_lengths.dtype,
        size=n).unstack(transcript_lengths)
    energy_ta = tf.TensorArray(
        dtype=energy.dtype,
        size=n).unstack(energy)

    def body(i, arr: tf.TensorArray):
        ul_t = utterance_lengths_ta.read(i)
        tl_t = transcript_lengths_ta.read(i)
        energy_t = energy_ta.read(i)
        attn = tf.nn.softmax(energy_t[:ul_t, :tl_t], axis=-1)
        pattn = tf.pad(attn, [[0, ul - ul_t], [0, tl - tl_t]])
        return i + 1, arr.write(i, pattn)

    loop_vars_in = (
        tf.constant(0, dtype=tf.int32),
        tf.TensorArray(size=n, dtype=utterance_h.dtype)
    )
    loop_vars_out = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=loop_vars_in
    )
    attn = loop_vars_out[1].stack()
    return attn


def attention_fn_softmax(utterance_h, utterance_mask, transcript_h, transcript_mask, dim):
    """

    (n, ul, d) * (n, d, tl) = (n, ul, tl)


    :param utterance_h: (ul, n, d)
    :param utterance_mask:(ul, n)
    :param transcript_h: (tl, n, d)
    :param transcript_mask:(tl, n)
    :param dim: key dimension
    :return:  # (n, ul, tl)
    """
    utterance_key = slim.fully_connected(
        utterance_h,
        num_outputs=dim,
        activation_fn=None,
        scope='utterance_attn')  # (ul, n, d)
    transcript_key = slim.fully_connected(
        transcript_h,
        num_outputs=dim,
        activation_fn=None,
        scope='transcript_attn')  # (tl, n, d)
    # align = tf.reduce_sum(tf.expand_dims(utterance_h, 1) * tf.expand_dims(transcript_h, 0), axis=-1)  # (ul, tl, n)
    energy = tf.matmul(
        tf.transpose(utterance_key, (1, 0, 2)),  # (n, ul, d)
        tf.transpose(transcript_key, (1, 2, 0))  # (n, d, tl)
    )  # (n, ul, tl)

    mask = tf.logical_and(
        tf.expand_dims(tf.transpose(utterance_mask, (1, 0)), 2),  # (n, ul, 1)
        tf.expand_dims(tf.transpose(transcript_mask, (1, 0)), 1)  # (n, 1, tl)
    )  # ( n, ul, tl)

    energy_idx = tf.where(mask)  # (nonzero, 3)
    energy_packed = tf.gather_nd(params=energy, indices=energy_idx)
    assert energy_packed.shape.ndims == 1
    assert energy_idx.shape.ndims == 2
    print("Energy: {}, {}".format(energy, energy.shape))
    energy_shape = tf.shape(energy, out_type=tf.int64)
    energy_sparse = tf.SparseTensor(
        values=energy_packed,
        indices=energy_idx,
        dense_shape=energy_shape
    )  # (n, ul, tl)
    attention_sparse = tf.sparse_softmax(energy_sparse)
    attention = sparse_tensor_to_dense_scatter(attention_sparse)
    return attention


def attention_fn_softmax_constlen(utterance_h, transcript_h, dim):
    """

    (n, ul, d) * (n, d, tl) = (n, ul, tl)


    :param utterance_h: (ul, n, d)
    :param utterance_mask:(ul, n)
    :param transcript_h: (tl, n, d)
    :param transcript_mask:(tl, n)
    :param dim: key dimension
    :return:  # (n, ul, tl)
    """
    utterance_key = slim.fully_connected(
        utterance_h,
        num_outputs=dim,
        activation_fn=None,
        scope='utterance_attn')  # (ul, n, d)
    transcript_key = slim.fully_connected(
        transcript_h,
        num_outputs=dim,
        activation_fn=None,
        scope='transcript_attn')  # (tl, n, d)
    # align = tf.reduce_sum(tf.expand_dims(utterance_h, 1) * tf.expand_dims(transcript_h, 0), axis=-1)  # (ul, tl, n)
    energy = tf.matmul(
        tf.transpose(utterance_key, (1, 0, 2)),  # (n, ul, d)
        tf.transpose(transcript_key, (1, 2, 0))  # (n, d, tl)
    )  # (n, ul, tl)
    attention = tf.nn.softmax(energy, axis=-1)
    return attention


def attention_fn_softplus(utterance_h, utterance_mask, transcript_h, transcript_mask, dim):
    """

    (n, ul, d) * (n, d, tl) = (n, ul, tl)


    :param utterance_h: (ul, n, d)
    :param utterance_mask:(ul, n)
    :param transcript_h: (tl, n, d)
    :param transcript_mask:(tl, n)
    :param dim: key dimension
    :return:  # (n, ul, tl)
    """
    utterance_h = slim.fully_connected(
        utterance_h,
        num_outputs=dim,
        activation_fn=None,
        scope='utterance_attn')  # (ul, n, d)
    # normalize utterance
    # utterance_h = (utterance_h /
    #               tf.sqrt(tf.maximum(1e-6, tf.reduce_sum(tf.square(utterance_h), axis=2, keepdims=True))))
    transcript_h = slim.fully_connected(
        transcript_h,
        num_outputs=dim,
        activation_fn=None,
        scope='transcript_attn')  # (tl, n, d)
    # normalize transcript
    # transcript_h = (transcript_h /
    #                tf.sqrt(tf.maximum(1e-6, tf.reduce_sum(tf.square(transcript_h), axis=2, keepdims=True))))
    # align = tf.reduce_sum(tf.expand_dims(utterance_h, 1) * tf.expand_dims(transcript_h, 0), axis=-1)  # (ul, tl, n)
    energy = tf.matmul(
        tf.transpose(utterance_h, (1, 0, 2)),  # (n, ul, d)
        tf.transpose(transcript_h, (1, 2, 0))  # (n, d, tl)
    )  # (n, ul, tl)

    mask = tf.logical_and(
        tf.expand_dims(tf.transpose(utterance_mask, (1, 0)), 2),  # (n, ul, 1)
        tf.expand_dims(tf.transpose(transcript_mask, (1, 0)), 1)  # (n, 1, tl)
    )  # ( n, ul, tl)

    energy = tf.nn.softplus(energy)
    energy_idx = tf.where(mask)  # (nonzero, 3)
    energy_shape = tf.shape(energy, out_type=tf.int64)

    # assert energy_packed.shape.ndims == 1
    assert energy_idx.shape.ndims == 2
    # print("Energy: {}, {}".format(energy, energy.shape))
    energy_sparse = tf.SparseTensor(
        values=tf.gather_nd(params=energy, indices=energy_idx),
        indices=energy_idx,
        dense_shape=energy_shape
    )  # (n, ul, tl)
    sum_energy = tf.sparse_reduce_sum(sp_input=energy_sparse, axis=2, keepdims=True)  # (n, ul, 1)
    sum_energy.set_shape([None, None, 1])

    attn = energy / tf.maximum(1., sum_energy)
    attn_sparse = tf.SparseTensor(
        values=tf.gather_nd(params=attn, indices=energy_idx),
        indices=energy_idx,
        dense_shape=energy_shape)
    attn = tf.sparse_tensor_to_dense(attn_sparse)
    # with tf.control_dependencies([tf.assert_near(tf.reduce_sum(attn[0,0,:]), 1)]):
    #    attn = tf.identity(attn)
    return attn
