import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

from asr_vae.kaldi.kaldi_records import write_records_app, write_records_flags


def main(_):
    dirs = [
        'train_si284',
        'test_dev93',
        'test_eval92',
        'test_eval93'
    ]
    write_records_app(dirs)


if __name__ == '__main__':
    write_records_flags(
        input_dir='../../data/wsj/export-fbank80-logpitch',
        data_dir='../../data/wsj/tfrecords-fbank80-logpitch-cmvn-global',
        files_per_shard=100,
        feats_file='feats-cmvn-global.ark',
        sentencepiece=False,
        vocab_size=100
    )
    tf.app.run()

"""
Dir: train_si284
Utterances: 37416
Shards: 374
Dir: test_dev93
Utterances: 503
Shards: 5
Dir: test_eval92
Utterances: 333
Shards: 3
Dir: test_eval93
Utterances: 213
Shards: 2
"""
