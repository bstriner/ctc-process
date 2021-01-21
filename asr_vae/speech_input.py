import os
from glob import glob

import numpy as np
import tensorflow as tf

from .mel import NUM_MEL_BINS


def decode_array(a, ashape, dtype):
    return tf.reshape(tf.decode_raw(a, dtype), ashape)


def speech_input(path, shuffle=True):
    search = os.path.join(path, '**', '*.tfrecord')
    print("Searching {}".format(search))
    paths = list(glob(search, recursive=True))
    assert len(paths) > 0
    feature = {
        'mel_feat': tf.FixedLenFeature([], tf.string),
        'mel_size': tf.FixedLenFeature([2], tf.int64),
        'transcript_feat': tf.FixedLenFeature([], tf.string),
        'transcript_size': tf.FixedLenFeature([1], tf.int64)
    }
    filename_queue = tf.train.string_input_producer(paths, num_epochs=None if shuffle else 1, shuffle=shuffle)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features=feature)

    mel = decode_array(features['mel_feat'], features['mel_size'], tf.float32)
    mel.set_shape([None, NUM_MEL_BINS])
    transcript = decode_array(features['transcript_feat'], features['transcript_size'], tf.int32)
    return mel, tf.shape(mel)[0], transcript, tf.shape(transcript)[0]


def speech_input_batch(path, shuffle=True):
    data_single = speech_input(path, shuffle=shuffle)
    dtypes = [tf.float32, tf.int32, tf.int32, tf.int32]
    if shuffle:
        queue = tf.RandomShuffleQueue(
            capacity=tf.flags.FLAGS.capacity,
            min_after_dequeue=tf.flags.FLAGS.min_after_dequeue,
            dtypes=dtypes)
    else:
        queue = tf.FIFOQueue(
            capacity=tf.flags.FLAGS.capacity,
            dtypes=dtypes
        )
    enqueue_op = queue.enqueue(data_single)
    qr = tf.train.QueueRunner(queue, [enqueue_op] * tf.flags.FLAGS.queue_threads)
    tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, qr)
    data_shuffled = queue.dequeue()
    for a, b in zip(data_shuffled, data_single):
        a.set_shape(b.get_shape())
    data_batch = tf.train.batch(
        data_shuffled,
        batch_size=tf.flags.FLAGS.batch_size,
        capacity=tf.flags.FLAGS.capacity,
        dynamic_pad=True,
        allow_smaller_final_batch=shuffle,
        name='shuffled_batch')

    utterances, utterance_lengths, transcripts, transcript_lengths = data_batch
    kw = {
        'utterances': utterances,
        'utterance_lengths': utterance_lengths,
        'transcripts': transcripts,
        'transcript_lengths': transcript_lengths
    }
    return kw, None


def speech_input_fns(base_path):
    def train_fn():
        return speech_input_batch(os.path.join(base_path, 'train-clean-360'))

    def eval_fn():
        return speech_input_batch(os.path.join(base_path, 'dev-clean'))

    def test_fn():
        return speech_input_batch(os.path.join(base_path, 'test-clean'), shuffle=False)

    vocab = np.load(os.path.join(base_path, 'train-clean-360', 'vocab.npy'))
    return train_fn, eval_fn, test_fn, vocab
