import tensorflow as tf


def generated_repeat(
        repeats,
        scope,
        name_scope,
        generation_fn,
        pack_idx,
        length,
        lengths,
        n,
        vocab_size
):
    # Generated
    generated = []
    for i in range(repeats):
        with tf.variable_scope(scope, reuse=True):
            with tf.name_scope('{}_{}'.format(name_scope, i)):
                logits_gen_packed = generation_fn()
                logits_gen = tf.scatter_nd(
                    indices=pack_idx,
                    updates=logits_gen_packed,
                    shape=(length, n, vocab_size + 1))
                generated_sparse, _ = tf.nn.ctc_greedy_decoder(
                    inputs=logits_gen,
                    sequence_length=lengths
                )
                generated.append(tf.sparse_tensor_to_dense(generated_sparse[0], default_value=-1))
    maxlen = tf.reduce_max([tf.shape(g)[1] for g in generated])
    generated = [tf.pad(g, paddings=[[0, 0], [0, maxlen - tf.shape(g)[1]]], constant_values=-1) for g in generated]
    generated = tf.stack(generated, axis=1)  # (n, repeats, maxlen)
    return generated
