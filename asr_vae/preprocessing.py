import os
import re
from glob import glob

import numpy as np
import soundfile as sf
import tensorflow as tf
import tqdm
from tensorflow.core.example.example_pb2 import Example
from tensorflow.core.example.feature_pb2 import Features, Feature
from tensorflow.python.lib.io.tf_record import TFRecordWriter

from .mel import calc_mel, SAMPLE_RATE


def trans_files(input_dir):
    search = os.path.join(input_dir, "**", "*.trans.txt")
    return glob(search, recursive=True)


def file_list(input_dir):
    trans = trans_files(input_dir)
    for t in trans:
        with open(t, 'r') as f:
            for l in f:
                l = l.strip()
                if len(l) > 0:
                    m = re.match(r'([\d\-]+) (.*)', l)
                    fn = m.group(1)
                    transcript = m.group(2)
                    ffn = os.path.join(os.path.dirname(t), '{}.flac'.format(fn))
                    assert os.path.exists(ffn)
                    yield ffn, transcript


def calc_vocab(files):
    v = set()
    for _, t in files:
        for c in t:
            v.add(c)
    v = list(v)
    v.sort()
    return v


def map_files(files, vocab):
    charmap = {c: i for i, c in enumerate(vocab)}
    for f, t in files:
        yield f, np.array([charmap[c] for c in t], dtype=np.int32)


def shard_files(n, shard_size):
    idx = np.arange(n)
    np.random.shuffle(idx)
    shard_count = - ((-n) // shard_size)
    shards = [idx[i * shard_size:min((i + 1) * shard_size, n)] for i in range(shard_count)]
    return shards


def preprocess(input_dir, output_dir, shard_size=100, vocab=None):
    os.makedirs(output_dir, exist_ok=True)
    print("Processing {}".format(input_dir))
    files = list(file_list(input_dir))
    if vocab is None:
        vocab = calc_vocab(files)
        np.save(os.path.join(output_dir, 'vocab.npy'), np.array(vocab, dtype=np.unicode_))
    print("Vocab size {}".format(vocab))
    files = list(map_files(files, vocab))
    n = len(files)
    print("File count: {}".format(n))
    shards = shard_files(n, shard_size=shard_size)
    audio_placeholder = tf.placeholder(dtype=tf.float32, name='audio', shape=(None,))
    mel_tensor = calc_mel(audio_placeholder)
    with tf.Session() as sess:
        prog = tqdm.tqdm(total=n, desc='Preprocessing')
        for i, shard in enumerate(shards):
            shard_file = os.path.join(output_dir, 'shard-{:012d}.tfrecord'.format(i))
            os.makedirs(os.path.dirname(shard_file), exist_ok=True)
            with TFRecordWriter(shard_file) as writer:
                for idx in shard:
                    file, transcript = files[idx]
                    audio, rate = sf.read(file=file)
                    assert len(audio.shape) == 1
                    assert rate == SAMPLE_RATE
                    mel = sess.run(mel_tensor, feed_dict={audio_placeholder: audio})
                    mel_feat, mel_size = feature_array(mel.astype(np.float32))
                    transcript_feat, transcript_size = feature_array(transcript.astype(np.int32))
                    feature = {
                        'mel_feat': mel_feat,
                        'mel_size': mel_size,
                        'transcript_feat': transcript_feat,
                        'transcript_size': transcript_size
                    }
                    example = Example(features=Features(feature=feature))
                    writer.write(example.SerializeToString())
                    prog.update(1)
        prog.close()
    return vocab


def feature_float32(value):
    return Feature(float_list=tf.train.FloatList(value=value.flatten()))


def feature_int64(value):
    return Feature(int64_list=tf.train.Int64List(value=value.flatten()))


def feature_string(value):
    binary = value.encode('utf-8')
    return Feature(bytes_list=tf.train.BytesList(value=[binary]))


def feature_bytes(value):
    return Feature(bytes_list=tf.train.BytesList(value=[value]))


def feature_array(arr):
    shp = np.array(arr.shape, np.int32)
    return feature_bytes(arr.flatten().tobytes()), feature_int64(shp)
