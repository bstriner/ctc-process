import glob
import os

import numpy as np

from .kaldi.transcripts import decode_transcript


def calc_word_vocab(transcript_dir, vocab):
    words = set()
    for f in glob.glob(os.path.join(transcript_dir, '**', '*.npy'), recursive=True):
        transcript = np.load(f)
        transcript_text = decode_transcript(transcript=transcript, vocab=vocab)
        for word in transcript_text.split(" "):
            words.add(word)
    words = list(words)
    words.sort()
    words = np.array(words, dtype=np.unicode_)
    return words


def calc_word_vocab_cached(transcript_dir, vocab, path):
    if os.path.exists(path):
        return np.load(path)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        words = calc_word_vocab(transcript_dir=transcript_dir, vocab=vocab)
        np.save(path, words)
        return words
