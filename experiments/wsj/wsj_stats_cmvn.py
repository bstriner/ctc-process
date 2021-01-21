import csv

import tensorflow as tf
import tqdm

from asr_vae.kaldi.inputs import make_input_fn
import numpy as np
import json


def main(_):
    # Test Data
    input_fn = make_input_fn(
        tf.app.flags.FLAGS.train_data_dir,
        shuffle=False,
        num_epochs=1,
        batch_size=1
    )
    ds = input_fn()
    it1 = ds.make_initializable_iterator()
    feats, labels = it1.get_next()
    print(feats)
    features = tf.squeeze(feats['features'], 0)
    featurelen = tf.squeeze(feats['feature_length'], 0)
    labellen = tf.squeeze(feats['label_length'], 0)
    dim = features.shape[-1].value
    print("DIM: {}".format(dim))
    running_sum = tf.get_local_variable(
        name='running_sum',
        dtype=tf.float32,
        shape=[dim],
        initializer=tf.initializers.zeros
    )
    running_count = tf.get_local_variable(
        name='running_count',
        dtype=tf.float32,
        shape=[],
        initializer=tf.initializers.zeros
    )
    update_sum = tf.assign_add(running_sum, tf.reduce_sum(features, axis=0))
    update_count = tf.assign_add(running_count, tf.cast(featurelen, tf.float32))
    update_mu = tf.group([update_sum, update_count])
    mu = running_sum / running_count

    ds = input_fn()
    it2 = ds.make_initializable_iterator()
    feats, labels = it2.get_next()
    features = tf.squeeze(feats['features'], 0)
    featurelen = tf.squeeze(feats['feature_length'], 0)
    labellen = tf.squeeze(feats['label_length'], 0)
    running_variance = tf.get_local_variable(
        name='running_variance',
        dtype=tf.float32,
        shape=[dim],
        initializer=tf.initializers.zeros
    )
    update_variance = tf.assign_add(running_variance, tf.reduce_sum(tf.squared_difference(features, mu), axis=0))
    variance = running_variance / running_count
    sigma = tf.sqrt(variance)

    with tf.train.MonitoredSession() as sess:
        sess.run(it1.initializer)
        sess.run(it2.initializer)
        prog = tqdm.tqdm(desc='mu')
        try:
            while True:
                sess.run(update_mu)
                prog.update(1)
        except tf.errors.OutOfRangeError as ex:
            prog.close()

        prog = tqdm.tqdm(desc='sigma')
        try:
            while True:
                sess.run(update_variance)
                prog.update(1)
        except tf.errors.OutOfRangeError as ex:
            prog.close()

        npmu, npsigma = sess.run([mu, sigma])
        np.savez(
            'wsj_stats_cmvn.npz',
            mu=npmu,
            sigma=npsigma
        )
        np.savetxt(
            'wsj_stats_cmvn-mu.txt',
            np.expand_dims(npmu, 1)
        )
        np.savetxt(
            'wsj_stats_cmvn-sigma.txt',
            np.expand_dims(npsigma, 1)
        )


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('train_data_dir', '../../data/wsj/tfrecords-fbank80-logpitch-cmvn-global/train_si284',
                               'Data directory')
    tf.app.flags.DEFINE_string('data_config', '../../data/wsj/tfrecords-fbank80-logpitch-cmvn-global/data_config.json',
                               'data_config file')
    tf.app.flags.DEFINE_integer('shuffle_buffer_size', 1000, 'shuffle_buffer_size')
    tf.app.flags.DEFINE_integer('prefetch_buffer_size', 100, 'prefetch_buffer_size')
    tf.app.flags.DEFINE_integer('num_parallel_calls', 4, 'num_parallel_calls')
    tf.app.run()
