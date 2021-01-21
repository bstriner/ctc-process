import os

import tensorflow as tf

from asr_vae.kaldi.transcripts import extract_vocab


def main(argv):
    file = os.path.join(tf.flags.FLAGS.kaldi_data_dir, 'train_clean_100', 'text')
    extract_vocab(file, os.path.join(tf.flags.FLAGS.data_dir, 'vocab.npy'))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('kaldi_data_dir', '/mnt/data/projects/kaldi/egs/librispeech/s5/data', 'Model directory')
    tf.flags.DEFINE_string('data_dir', '../../data/librispeech', 'Model directory')
    tf.app.run()
