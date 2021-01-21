import json
import os

import kaldiio
import numpy as np
import sentencepiece
import tensorflow as tf
import tqdm

from .constants import *
from .record_writer import ShardRecordWriter
from .text_normalization import normalize_sentence_for_training

FLAGS = tf.app.flags.FLAGS


def audio_sequence_example(features, labels, speaker_id, utterance_id, raw_text):
    return tf.train.SequenceExample(
        context=tf.train.Features(
            feature={
                FEATURE_LENGTH: tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=[features.shape[0]]
                    )
                ),
                LABEL_LENGTH: tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=[labels.shape[0]]
                    )
                ),
                SPEAKER_ID: tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[speaker_id.encode('utf-8')]
                    )
                ),
                UTTERANCE_ID: tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[utterance_id.encode('utf-8')]
                    )
                ),
                RAW_TEXT: tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[raw_text.encode('utf-8')]
                    )
                ),
            }
        ),
        feature_lists=tf.train.FeatureLists(
            feature_list={
                FEATURES: tf.train.FeatureList(
                    feature=[
                        tf.train.Feature(float_list=tf.train.FloatList(value=f))
                        for f in features
                    ]
                ),
                LABELS: tf.train.FeatureList(
                    feature=[
                        tf.train.Feature(int64_list=tf.train.Int64List(value=[l]))
                        for l in labels
                    ]
                )
            }
        )
    )


def read_transcripts(transcripts_path):
    with open(transcripts_path, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                a = line.split(" ")
                yield a[0], " ".join(a[1:])


def read_speakers(utt2spk_path):
    with open(utt2spk_path, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                a = line.split(" ")
                assert len(a) == 2
                yield a[0], a[1]


def write_transcripts(transcripts_path, transcripts):
    with open(transcripts_path, 'w') as f:
        for key, text in transcripts:
            f.write("{} {}\n".format(key, text))


def read_ark(path):
    stack = []
    id = None
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                line = line.split(" ")
                if line[-1] == '[':
                    assert len("".join(line[1:-1])) == 0
                    id = line[0]
                elif line[-1] == ']':
                    data = [float(d) for d in line[:-1]]
                    stack.append(data)
                    yield id, np.array(stack, dtype=np.float32)
                    id = None
                    stack = []
                else:
                    data = [float(d) for d in line]
                    stack.append(data)


def make_examples(features_path,
                  transcripts_raw,
                  transcripts_encoded, speakers):
    # transcripts = {k: v for k, v in read_transcripts(transcripts_path)}
    for id, features in tqdm.tqdm(kaldiio.load_ark(features_path), total=len(transcripts_raw)):
        features = features.astype(np.float32)
        transcript_raw = transcripts_raw[id]
        transcript_encoded = transcripts_encoded[id]
        if len(transcript_encoded) > 0:
            speaker = speakers[id]
            yield audio_sequence_example(
                features=features,
                labels=transcript_encoded,
                speaker_id=speaker,
                utterance_id=id,
                raw_text=transcript_raw)
        else:
            print("Skipping empty transcript {}".format(id))


def write_records(
        features_path, records_path, files_per_shard,
        transcripts_raw,
        transcripts_encoded,
        speakers
):
    os.makedirs(records_path, exist_ok=True)
    path_fmt = os.path.join(records_path, 'records-{:08d}.tfrecords')
    with ShardRecordWriter(path_fmt=path_fmt, chunksize=files_per_shard) as tfr:
        examples = make_examples(
            features_path=features_path,
            transcripts_raw=transcripts_raw,
            transcripts_encoded=transcripts_encoded,
            speakers=speakers)
        for example in examples:
            tfr.write(example.SerializeToString())


def calc_vocab(all_transcripts):
    vocab = set()
    for ds in all_transcripts:
        for id, transcript in ds.items():
            vocab.update(transcript)
    vocab = list(vocab)
    vocab.sort()
    vocab = np.array(vocab, dtype=np.unicode_)
    return vocab


def encode_transcripts(transcripts, vocab_map):
    for id, transcript in transcripts.items():
        yield id, np.array([vocab_map[c] for c in transcript], dtype=np.int64)


def clean_piece(piece: str):
    return piece.replace(chr(9601), " ")


def write_records_app(dirs):
    data_dir = tf.app.flags.FLAGS.data_dir
    input_dir = tf.app.flags.FLAGS.input_dir
    files_per_shard = tf.app.flags.FLAGS.files_per_shard
    sentencepiece_enabled = tf.app.flags.FLAGS.sentencepiece
    os.makedirs(data_dir, exist_ok=True)
    all_transcripts_raw = []
    all_speakers = []
    for d in tqdm.tqdm(dirs, desc='Transcripts'):
        transcripts_path = os.path.join(input_dir, d, 'text')
        speakers_path = os.path.join(input_dir, d, 'utt2spk')
        all_transcripts_raw.append(dict(read_transcripts(transcripts_path)))
        all_speakers.append(dict(read_speakers(speakers_path)))

    all_transcripts_normalized = [
        {k: normalize_sentence_for_training(v) for k, v in ts.items()} for ts in all_transcripts_raw
    ]

    train_features_path = os.path.join(input_dir, dirs[0], FLAGS.feats_file)
    for id, features in kaldiio.load_ark(train_features_path):
        input_dim = features.shape[-1]
        break

    if sentencepiece_enabled:
        sp_train = os.path.join(data_dir, 'sentencepiece-train.txt')
        sp_prefix = os.path.join(data_dir, 'sentencepiece-model')
        sp_model = "{}.model".format(sp_prefix)
        vocab_size = tf.app.flags.FLAGS.vocab_size
        opts = [
            "--input={}".format(sp_train),
            "--vocab_size={}".format(vocab_size),
            "--character_coverage=1.0",
            "--model_type=unigram",
            "--model_prefix={}".format(sp_prefix),
        ]
        opts = " ".join(opts)
        print("Opts: {}".format(opts))
        with open(sp_train, 'w') as fp:
            for t in all_transcripts_normalized[0].values():
                fp.write(t + "\n")
        sentencepiece.SentencePieceTrainer.Train(opts)
        sp = sentencepiece.SentencePieceProcessor()
        sp.Load(sp_model)
        vocab = [clean_piece(sp.IdToPiece(i)) for i in range(vocab_size)]
        vocab = np.array(vocab, dtype=np.unicode)
        all_transcripts_encoded = [
            {k: np.array(sp.EncodeAsIds(v), dtype=np.int64) for k, v in ts.items()}
            for ts in all_transcripts_normalized
        ]
    else:
        vocab = calc_vocab(all_transcripts_normalized)
        vocab_map = {v: i for i, v in enumerate(vocab)}
        vocab_size = len(vocab)
        print("Vocab: {}".format(vocab))
        all_transcripts_encoded = [
            {k: np.array([vocab_map[c] for c in v], dtype=np.int64) for k, v in ts.items()}
            for ts in all_transcripts_normalized
        ]
    np.save(os.path.join(data_dir, VOCAB), vocab)

    print("Feature dim: {}".format(input_dim))
    info = {
        'input_dim': input_dim,
        'vocab': list(vocab),
        'sentencepiece': sentencepiece_enabled,
        'vocab_size': vocab_size,
        'data_dir': data_dir,
        'input_dir': input_dir,
        'files_per_shard': files_per_shard
    }
    with open(os.path.join(data_dir, DATA_CONFIG), 'w', encoding="utf-8") as fp:
        json.dump(info, fp, indent=4, sort_keys=True)

    it = tqdm.tqdm(zip(dirs, all_transcripts_raw, all_transcripts_encoded, all_speakers), desc='Dataset')
    for d, transcripts_raw, transcripts_encoded, speakers in it:
        print("Dir: {}".format(d))
        features_path = os.path.join(input_dir, d, FLAGS.feats_file)
        records_path = os.path.join(data_dir, d)
        write_records(
            transcripts_raw=transcripts_raw,
            transcripts_encoded=transcripts_encoded,
            speakers=speakers,
            features_path=features_path,
            records_path=records_path,
            files_per_shard=files_per_shard
        )


def write_records_flags(
        input_dir, data_dir, files_per_shard,
        feats_file='feats.ark', sentencepiece=False, vocab_size=2000):
    tf.app.flags.DEFINE_string('input_dir', input_dir, 'Data directory')
    tf.app.flags.DEFINE_string('data_dir', data_dir, 'Model directory')
    tf.app.flags.DEFINE_integer('files_per_shard', files_per_shard, 'Data directory')
    tf.app.flags.DEFINE_integer('vocab_size', vocab_size, 'vocab_size')
    tf.app.flags.DEFINE_string('feats_file', feats_file, 'features_file')
    tf.app.flags.DEFINE_bool('sentencepiece', sentencepiece, 'sentencpiece')
