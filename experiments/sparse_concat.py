import tensorflow as tf

vocab = [" "]+[chr(ord('a')+i) for i in range(26)]
vocab = tf.constant(vocab, dtype=tf.strin)
table = tf.contrib.lookup.index_to_string_table_from_tensor(
    vocab, default_value="")
values = table.lookup(vocab)
