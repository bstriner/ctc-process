import itertools

import numpy as np
import tensorflow as tf

from asr_vae.models.metrics.asr_metrics import asr_metrics_and_summaries, strings_to_chars, strings_to_words


def text_to_sparse_tensor(text, charmap):
    true_ids = [[charmap[v] for v in t] for t in text]
    true_maxchar = max(len(t) for t in text)
    true_idx = np.array(list(itertools.chain.from_iterable(
        zip([i] * len(text[i]), range(len(text[i])))
        for i in range(len(text))
    )), dtype=np.int64)
    true_values = np.array(list(itertools.chain.from_iterable(true_ids)), dtype=np.int64)
    sparse = tf.SparseTensor(
        dense_shape=(len(text), true_maxchar),
        indices=true_idx,
        values=true_values
    )
    return sparse


if __name__ == '__main__':
    """
    x = 'the quick brown fox'
    x = list(x)
    print(x)

    xt = tf.constant(np.array(x, dtype=np.unicode_), dtype=tf.string)
    xj = tf.reduce_join(xt)
    with tf.train.MonitoredSession() as s:
        print(s.run(xt))
        print(s.run(xj))
    """

    vocab = np.array([' '] + [chr(i) for i in range(ord('a'), ord('z') + 1)], dtype=np.unicode_)
    print(len(vocab))
    charmap = {v: i for i, v in enumerate(vocab)}

    vocab_tensor = tf.constant(vocab, dtype=tf.string)
    print(vocab_tensor)

    true_text = [
        'hello world',
        'the quick brown fox jumped over the lazy brown dog'
    ]

    hyp_text = [
        'hello world baby',
        'the slow brown fox'
    ]
    eval_metric_ops = {}
    predictions = {}
    asr_metrics_and_summaries(
        transcripts_strings=true_text,
        transcripts_sparse=None,
        decoded_strings=hyp_text,
        decoded_sparse=None,
        eval_metric_ops=eval_metric_ops,
        prefix="",
        predictions=predictions,
        sentencepiece=False)
    print(predictions)
    import pprint

    a = strings_to_words(true_text)
    b = strings_to_chars(true_text)
    with tf.train.MonitoredSession() as sess:
        preds = sess.run(predictions)
        pprint.pprint(preds)
        pprint.pprint(sess.run(a))
        pprint.pprint(sess.run(b))
    #    for k, v in metrics.items():
    #        print("{}: {}".format(k, sess.run(v[0])))

    """

    true_chars = sparse_to_dense_chars(true_sparse, vocab_table)
    true_strings = reduce_join(true_chars, axis=1)
    true_words = string_split(true_strings)
    # true_wordcounts = strings_to_wordcounts(true_strings)

    vocab_table=tf.contrib.lookup.index_to_string_table_from_tensor(
        vocab_tensor,
        default_value='',
        name='vocab_table'
    )
    true_chars = sparse_to_dense_chars(true_sparse, vocab_table)
    true_strings = characters_to_strings(true_chars)
    true_wordcounts = strings_to_wordcounts(true_strings)

    hypothesis_chars = sparse_to_dense_chars(hypothesis_sparse, vocab_table)
    hypothesis_strings = characters_to_strings(hypothesis_chars)
    hypothesis_wordcounts = strings_to_wordcounts(hypothesis_strings)

    with 
    pass
    """
