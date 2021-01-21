import os

import tensorflow as tf

from asr_vae.kaldi.features import extract_features


def main(_argv):
    dirs = [
        'train_clean_100',
        'dev_clean',
        'test_clean'
    ]
    for d in dirs:
        feats_file = os.path.join(tf.flags.FLAGS.kaldi_data_dir, d, 'feats-extract.ark')
        path = os.path.join(tf.flags.FLAGS.data_dir, d, 'features')
        extract_features(
            feats_file=feats_file,
            path=path)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('kaldi_data_dir', '/mnt/data/projects/kaldi/egs/librispeech/s5/data', 'Model directory')
    tf.flags.DEFINE_string('data_dir', '../../data/librispeech', 'Model directory')
    tf.app.run()
