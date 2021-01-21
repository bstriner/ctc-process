import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

from asr_vae.kaldi.kaldi_records import write_records_app, write_records_flags


def main(_):
    dirs = [
        'train_clean_360',
        'train_clean_100',
        'dev_clean',
        'test_clean'
    ]
    write_records_app(dirs)


if __name__ == '__main__':
    write_records_flags(
        input_dir='../../data/librispeech/textdata',
        data_dir='../../data/librispeech/tfrecords',
        files_per_shard=100
    )
    tf.app.run()

"""
Dir: train_clean_100, Files: 28539, Shards: 285
Dir: dev_clean, Files: 2703, Shards: 27
Dir: test_clean, Files: 2620, Shards: 26
"""