import os

import tensorflow as tf

from asr_vae.kaldi.transcripts import extract_transcripts


def main(_argv):
    dirs = [
        'train_clean_100',
        'dev_clean',
        'test_clean'
    ]
    vocab_file = os.path.join(tf.flags.FLAGS.data_dir, 'vocab.npy')
    for d in dirs:
        text_file = os.path.join(tf.flags.FLAGS.kaldi_data_dir, d, 'text')
        path = os.path.join(tf.flags.FLAGS.data_dir, d, 'transcripts')
        extract_transcripts(
            text_file=text_file,
            vocab_file=vocab_file,
            path=path)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('kaldi_data_dir', '/mnt/data/projects/kaldi/egs/librispeech/s5/data', 'Model directory')
    tf.flags.DEFINE_string('data_dir', '../../data/librispeech', 'Model directory')
    tf.app.run()
