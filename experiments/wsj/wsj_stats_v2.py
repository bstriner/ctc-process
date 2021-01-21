import csv
import os

import tensorflow as tf
import tqdm

from asr_vae.kaldi.inputs import make_input_fn


def main(_):
    # Test Data
    input_fn = make_input_fn(
        tf.app.flags.FLAGS.train_data_dir,
        shuffle=False,
        num_epochs=1,
        batch_size=1,
        subsample=1)
    ds = input_fn()
    it = ds.make_initializable_iterator()
    feats, labels = it.get_next()
    features = tf.squeeze(feats['features'], 0)
    featurelen = tf.squeeze(feats['feature_length'], 0)
    labels = tf.squeeze(feats['labels'], 0)
    labellen = tf.squeeze(feats['label_length'], 0)
    dim = features.shape[-1].value
    print("DIM: {}".format(dim))

    prog = tqdm.tqdm()
    total_utterance = 0
    total_transcript = 0
    output_file = tf.app.flags.FLAGS.output_file
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Id', 'Utterance Length', 'Transcript Length', 'Ratio'])
        with tf.train.MonitoredSession() as sess:
            sess.run(it.initializer)
            try:
                i = 0
                while True:
                    f, fl, l, ll = sess.run([features, featurelen, labels, labellen])
                    assert f.shape[0] == fl
                    assert l.shape[0] == ll
                    w.writerow([i, fl, ll, fl / ll])
                    prog.update(1)
                    i += 1
                    total_utterance += fl
                    total_transcript += ll
            except tf.errors.OutOfRangeError as ex:
                print("Done")

    prog.close()
    total_ratio = total_utterance / total_transcript
    print("Total Utterance: {}, Total Transcript: {}, Average Ratio: {}".format(
        total_utterance,
        total_transcript,
        total_ratio
    ))


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('train_data_dir', '../../data/wsj/tfrecords-mel-40-cmvn-speaker/train_si284',
                               'Data directory')
    tf.app.flags.DEFINE_string('data_config', '../../data/wsj/tfrecords-mel-40-cmvn-speaker/data_config.json',
                               'data_config file')
    tf.app.flags.DEFINE_string('output_file', '../../output/wsj/tfrecords-mel-40-cmvn-speaker-stats.csv',
                               'data_config file')
    tf.app.flags.DEFINE_integer('shuffle_buffer_size', 1000, 'shuffle_buffer_size')
    tf.app.flags.DEFINE_integer('prefetch_buffer_size', 100, 'prefetch_buffer_size')
    tf.app.flags.DEFINE_integer('num_parallel_calls', 4, 'num_parallel_calls')
    tf.app.run()
