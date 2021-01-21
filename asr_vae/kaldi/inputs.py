import glob
import os

import numpy as np
import sentencepiece as spm
import tensorflow as tf

from .constants import *
from .record_writer import COMPRESSION_TYPE_STR
from .spec_augment import augment
from .text_normalization import normalize_sentence_for_training


def sentencepiece_op_func(inp, sp: spm.SentencePieceProcessor, alpha=0.2):
    def func(x: bytes):
        sent = normalize_sentence_for_training(x.decode('utf-8'))
        sent = sp.SampleEncodeAsIds(sent, -1, alpha)
        sent = np.array(sent, np.int32)
        return sent

    y = tf.py_func(func=func, inp=[inp], Tout=tf.int32, stateful=True)
    y.set_shape([None])
    return y


def subsample_utterance(utterance, utterance_length, subsample, average, independent_subsample):
    if subsample > 1:
        new_len = (-(tf.math.floordiv(-utterance_length, subsample)))
        new_total_len = new_len * subsample
        input_dim = utterance.shape[-1].value
        assert input_dim

        utterance_features = tf.pad(utterance, [(0, new_total_len - utterance_length), (0, 0)])
        utterance_features = tf.reshape(utterance_features, (new_len, subsample, input_dim))
        if average:
            utterance = tf.reduce_mean(utterance_features, axis=1)
        else:
            if independent_subsample:
                idx0 = tf.range(start=0, limit=tf.cast(new_len, dtype=tf.int64), dtype=tf.int64)
                idx1 = tf.random.uniform(minval=0, maxval=subsample, dtype=tf.int64, shape=[new_len])
                idx = tf.stack([idx0, idx1], axis=-1)
                utterance = tf.gather_nd(indices=idx, params=utterance_features)
            else:
                idx = tf.random.uniform(minval=0, maxval=subsample, dtype=tf.int64, shape=[])
                utterance = utterance_features[:, idx, :]

            # v2
            # idx = tf.random.uniform(minval=0, maxval=subsample, dtype=tf.int64, shape=[])
            # utterance = utterance_features[:, idx, :]
        utterance_length = new_len
    return utterance, utterance_length


def make_parse_example(subsample, average, input_dim, independent_subsample, add_one=False,
                       sa_params=None, sp=None):
    def parse_example(serialized_example):
        context_features = {
            FEATURE_LENGTH: tf.io.FixedLenFeature([1], tf.int64),
            LABEL_LENGTH: tf.io.FixedLenFeature([1], tf.int64),
            SPEAKER_ID: tf.io.FixedLenFeature([1], tf.string),
            UTTERANCE_ID: tf.io.FixedLenFeature([1], tf.string),
            RAW_TEXT: tf.io.FixedLenFeature([1], tf.string)
        }
        sequence_features = {
            # FEATURES: tf.io.VarLenFeature(tf.float32),
            # LABELS: tf.io.VarLenFeature(tf.int64),
            FEATURES: tf.io.FixedLenSequenceFeature([input_dim], tf.float32),
            LABELS: tf.io.FixedLenSequenceFeature([1], tf.int64),
        }

        context_parse, sequence_parsed = tf.io.parse_single_sequence_example(
            serialized_example, context_features, sequence_features
        )
        feature_length = context_parse[FEATURE_LENGTH]
        feature_length = tf.squeeze(feature_length, -1)
        feature_length = tf.cast(feature_length, dtype=tf.int32)

        features = sequence_parsed[FEATURES]
        # features = tf.sparse_tensor_to_dense(features)
        print("sequence_parsed[FEATURES]: {}".format(sequence_parsed[FEATURES]))
        # features = tf.reshape(features, (-1, input_dim))
        raw_text = tf.squeeze(context_parse[RAW_TEXT], axis=-1)
        if sp is None:
            transcript = sequence_parsed[LABELS]
            # transcript = tf.sparse_tensor_to_dense(transcript)
            transcript = tf.squeeze(transcript, 1)
            transcript = tf.cast(transcript, dtype=tf.int32)
            transcript_length = context_parse[LABEL_LENGTH]
            transcript_length = tf.squeeze(transcript_length, -1)
            transcript_length = tf.cast(transcript_length, dtype=tf.int32)
        else:
            transcript = sentencepiece_op_func(
                inp=raw_text,
                sp=sp
            )
            transcript_length = tf.shape(transcript, out_type=tf.int32)[0]

        print("FL: {}, {}".format(feature_length, type(feature_length)))
        if sa_params is not None:
            features = augment(
                utterance=features,
                sa_params=sa_params
            )
        features, feature_length = subsample_utterance(
            features, feature_length, subsample=subsample,
            independent_subsample=independent_subsample, average=average)
        if add_one:
            transcript = transcript + 1
        feats = {
            FEATURES: features,
            FEATURE_LENGTH: feature_length,
            LABELS: transcript,
            LABEL_LENGTH: transcript_length,
            SPEAKER_ID: tf.squeeze(context_parse[SPEAKER_ID], axis=-1),
            UTTERANCE_ID: tf.squeeze(context_parse[UTTERANCE_ID], axis=-1),
            RAW_TEXT: raw_text
        }
        labels = {
            LABELS: transcript,
            LABEL_LENGTH: transcript_length
        }
        return feats, labels

    return parse_example


def dataset_single(filenames, input_dim, sa_params=None, num_epochs=1, add_one=False,
                   shuffle=True, subsample=1, independent_subsample=True,
                   average=False, sp=None):
    ds = tf.data.TFRecordDataset(filenames=filenames, compression_type=COMPRESSION_TYPE_STR)
    if shuffle:
        ds = ds.shuffle(tf.app.flags.FLAGS.shuffle_buffer_size)
    ds = ds.repeat(num_epochs)
    ds = ds.map(
        make_parse_example(
            subsample=subsample,
            average=average,
            input_dim=input_dim,
            independent_subsample=independent_subsample,
            sa_params=sa_params,
            sp=sp,
            add_one=add_one),
        num_parallel_calls=tf.app.flags.FLAGS.num_parallel_calls)
    return ds


def dataset_batch(ds_single: tf.data.Dataset, input_dim, batch_size=5):
    feat_shapes = {
        FEATURES: [None, input_dim],
        FEATURE_LENGTH: [],
        LABELS: [None],
        LABEL_LENGTH: [],
        RAW_TEXT: [],
        SPEAKER_ID: [],
        UTTERANCE_ID: [],
    }
    label_shapes = {
        LABELS: [None],
        LABEL_LENGTH: [],
    }
    # label_shapes = [None], []
    ds = ds_single.padded_batch(
        batch_size=batch_size,
        padded_shapes=(feat_shapes, label_shapes),
        drop_remainder=False)
    return ds


def get_file_list(path):
    return list(glob.glob(os.path.join(path, "**", "*.tfrecords"), recursive=True))


def make_input_fn(data_dir, batch_size, input_dim, sa_params=None, shuffle=True,
                  independent_subsample=True, num_epochs=None, add_one=False,
                  subsample=1, average=False, bucket=False, bucket_size=100, buckets=30, sp=None):
    def input_fn():
        ds = data_dir
        ds = get_file_list(ds)  # tf.data.Dataset.list_files(os.path.join(ds, '*.tfrecords'), shuffle=shuffle)
        ds = dataset_single(
            ds,
            input_dim=input_dim,
            shuffle=shuffle,
            num_epochs=num_epochs,
            subsample=subsample,
            independent_subsample=independent_subsample,
            average=average,
            sa_params=sa_params,
            sp=sp,
            add_one=add_one
        )
        if bucket:
            bucket_boundaries = list((i + 1) * bucket_size for i in range(buckets))
            bucket_batch_sizes = list(batch_size for i in range(buckets + 1))
            ds = tf.data.experimental.bucket_by_sequence_length(
                element_length_func=lambda features, labels: features[FEATURE_LENGTH],
                bucket_boundaries=bucket_boundaries,
                bucket_batch_sizes=bucket_batch_sizes,
                # padded_shapes=(feat_shapes, label_shapes),
                padding_values=None,
                pad_to_bucket_boundary=False,
                no_padding=False,
                drop_remainder=False
            )(ds)
        else:
            ds = dataset_batch(ds, input_dim=input_dim, batch_size=batch_size)
        ds = ds.prefetch(buffer_size=tf.app.flags.FLAGS.prefetch_buffer_size)
        return ds

    return input_fn
