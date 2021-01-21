import tensorflow as tf

from ...sparse import sparse_to_counts

EVAL_SUMMARIES = 'eval_summaries'
SLOW_SUMMARIES = 'slow_summaries'


def sparse_to_dense_chars(sparse: tf.SparseTensor, vocab_tensor):
    # char_values = vocab_table.lookup(tf.cast(
    #    sparse.values, tf.int64))
    char_values = tf.gather(
        params=vocab_tensor,
        indices=sparse.values,
        axis=0
    )
    char_sparse = tf.SparseTensor(
        values=char_values,
        indices=sparse.indices,
        dense_shape=sparse.dense_shape
    )
    dense_chars = tf.sparse_tensor_to_dense(
        char_sparse,
        default_value=tf.constant("", dtype=tf.string)
    )
    return dense_chars


def sparse_to_dense_strings(sparse, vocab_table):
    chars = sparse_to_dense_chars(sparse, vocab_table)
    return tf.strings.reduce_join(chars, axis=-1)


def sparse_to_clean_strings(sparse, vocab_table):
    strings = sparse_to_dense_strings(sparse, vocab_table)
    strings = tf.strings.regex_replace(
        input=strings,
        pattern=" +",
        rewrite=" ",
        replace_global=True,
        name=None
    )
    strings = tf.strings.strip(strings)
    return strings


def strings_to_chars(strings):
    return tf.string_split(strings, delimiter='')


def strings_to_words(strings):
    return tf.string_split(strings, delimiter=' ')


def make_vocab_table(vocab_tensor):
    return tf.contrib.lookup.index_to_string_table_from_tensor(
        vocab_tensor,
        default_value='',
        name='vocab_table'
    )


def asr_metrics(true_count, hypothesis_count, edit_distance, prefix):
    true_count_metric = tf.metrics.mean(true_count)
    hypothesis_count_metric = tf.metrics.mean(hypothesis_count)
    edit_distance_metric = tf.metrics.mean(edit_distance)
    error_rate_metric = ((edit_distance_metric[0]) / (true_count_metric[0])), tf.no_op()

    ops = {
        '{}edit_distance'.format(prefix): edit_distance_metric,
        '{}error_rate'.format(prefix): error_rate_metric,
        '{}count_true'.format(prefix): true_count_metric,
        '{}count_hypothesis'.format(prefix): hypothesis_count_metric
    }
    return ops


def asr_summaries(true_count, hypothesis_count, edit_distance, prefix, collections=None):
    true_count = tf.reduce_mean(true_count)
    hypothesis_count = tf.reduce_mean(hypothesis_count)
    edit_distance = tf.reduce_mean(edit_distance)
    error_rate = tf.cast(edit_distance, tf.float32) / tf.cast(true_count, tf.float32)

    tf.summary.scalar('{}edit_distance'.format(prefix), edit_distance, collections=collections)
    tf.summary.scalar('{}error_rate'.format(prefix), error_rate, collections=collections)
    tf.summary.scalar('{}count_true'.format(prefix), true_count, collections=collections)
    tf.summary.scalar('{}count_hypothesis'.format(prefix), hypothesis_count, collections=collections)


def asr_metric_and_summary(
        true_tokens,
        hypothesis_tokens,
        eval_metric_ops,
        prefix="",
        predictions=None
):
    true_count = sparse_to_counts(true_tokens)
    hypothesis_count = sparse_to_counts(hypothesis_tokens)

    edit_distance = tf.edit_distance(
        hypothesis=hypothesis_tokens,
        truth=true_tokens,
        normalize=False
    )
    eval_metric_ops.update(
        asr_metrics(
            true_count=true_count,
            hypothesis_count=hypothesis_count,
            edit_distance=edit_distance,
            prefix=prefix
        )
    )
    asr_summaries(
        true_count=true_count,
        hypothesis_count=hypothesis_count,
        edit_distance=edit_distance,
        prefix=prefix,
        collections=[SLOW_SUMMARIES]
    )
    if predictions is not None:
        rate = tf.cast(edit_distance, tf.float32) / tf.cast(true_count, tf.float32)
        predictions.update({
            prefix + "count_true": true_count,
            prefix + "count_hypothesis": hypothesis_count,
            prefix + "edit_distance": edit_distance,
            prefix + "error_rate": rate
        })


def asr_metrics_and_summaries(
        transcripts_strings,
        transcripts_sparse,
        decoded_strings,
        decoded_sparse,
        eval_metric_ops,
        prefix="",
        predictions=None,
        sentencepiece=False
):
    asr_metric_and_summary(
        true_tokens=strings_to_chars(transcripts_strings),
        hypothesis_tokens=strings_to_chars(decoded_strings),
        eval_metric_ops=eval_metric_ops,
        predictions=predictions,
        prefix=prefix + "character_"
    )
    asr_metric_and_summary(
        true_tokens=strings_to_words(transcripts_strings),
        hypothesis_tokens=strings_to_words(decoded_strings),
        eval_metric_ops=eval_metric_ops,
        predictions=predictions,
        prefix=prefix + "word_"
    )
    if sentencepiece:
        decoded_cast = tf.SparseTensor(
            values=tf.cast(decoded_sparse.values, dtype=transcripts_sparse.values.dtype),
            dense_shape=decoded_sparse.dense_shape,
            indices=decoded_sparse.indices
        )
        asr_metric_and_summary(
            true_tokens=transcripts_sparse,
            hypothesis_tokens=decoded_cast,
            eval_metric_ops=eval_metric_ops,
            predictions=predictions,
            prefix=prefix + "piece_"
        )

    #generated_string_table = tf.stack([transcripts_strings, decoded_strings], axis=1)
    #tf.summary.text(
    #    prefix + "generated_transcripts",
    #    generated_string_table,
    #    collections=[SLOW_SUMMARIES, EVAL_SUMMARIES]
    #)
