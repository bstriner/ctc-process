import tensorflow as tf


def check_inputs(utterances, utterance_lengths, transcripts, transcript_lengths, vocab_size, reduction=1):
    with tf.control_dependencies([
        tf.assert_positive(utterance_lengths)
    ]):
        utterance_lengths = tf.identity(utterance_lengths)
    with tf.control_dependencies([
        tf.assert_positive(transcript_lengths),
        tf.assert_greater_equal(tf.floor_div(utterance_lengths, reduction), transcript_lengths)
    ]):
        transcript_lengths = tf.identity(transcript_lengths)
    with tf.control_dependencies([
        tf.assert_greater_equal(transcripts, 0),
        tf.assert_less(transcripts, vocab_size)
    ]):
        transcripts = tf.identity(transcripts)
    return utterances, utterance_lengths, transcripts, transcript_lengths
