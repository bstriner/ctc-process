import os

import numpy as np


def read_transcripts(file):
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            chunks = line.split(" ")
            id = chunks[0]
            text = " ".join(chunks[1:])
            yield id, text


def make_charmap(vocab):
    return {v: i for i, v in enumerate(vocab)}


def map_transcript(text, charmap):
    return np.array([charmap[c] for c in text], dtype=np.int32)


def map_transcripts(transcripts, charmap):
    for id, text in transcripts:
        yield id, map_transcript(text, charmap)


def decode_transcript(transcript, vocab):
    return "".join(vocab[t] for t in transcript if t >= 0)


def write_transcripts(transcripts, path):
    os.makedirs(path, exist_ok=True)
    for id, arr in transcripts:
        fout = os.path.join(path, '{}.npy'.format(id))
        np.save(fout, arr)


def extract_vocab(file, vocab_file):
    os.makedirs(os.path.dirname(vocab_file), exist_ok=True)
    vocab = set()
    for id, text in read_transcripts(file):
        for char in text:
            vocab.add(char)
    vocab = list(vocab)
    vocab.sort()
    vocab = np.array(vocab, dtype=np.unicode_)
    np.save(vocab_file, vocab)
    print("Vocab size: {}".format(vocab.shape))
    print(vocab)


def extract_transcripts(text_file, vocab_file, path):
    vocab = np.load(vocab_file)
    charmap = make_charmap(vocab)
    transcripts = read_transcripts(text_file)
    transcripts = map_transcripts(transcripts, charmap)
    write_transcripts(transcripts, path)
