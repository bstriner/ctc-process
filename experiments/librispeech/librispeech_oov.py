import csv
import os

import numpy as np

from asr_vae.oov import calc_word_vocab_cached


def csv_wordlist(file, col):
    words = set()
    with open(file, 'r') as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row)>col:
                for word in row[col].split(" "):
                    words.add(word)
    words = list(words)
    words.sort()
    return words


if __name__ == '__main__':
    data_dir = '../../data/librispeech'
    vocab = np.load(os.path.join(data_dir, 'vocab.npy'))
    output_dir = '../../output/librispeech/oov'
    dirs = [
        'train_clean_100',
        'dev_clean',
        'test_clean'
    ]
    wordsets = []
    for d in dirs:
        transcript_dir = os.path.join(data_dir, d, "transcripts")
        output_path = os.path.join(output_dir, "{}.npy".format(d))
        words = calc_word_vocab_cached(
            transcript_dir=transcript_dir,
            vocab=vocab,
            path=output_path
        )
        wordsets.append(words)
        print(words.size)
        print(words[:10])

    #eval_file = "../../output/bak/asr_vae_ctc_cudnn/v31-2depth-256dim-l21e-4-min1e-2/generated/iteration-000000052000.csv"
    #eval_file =  r"D:\Projects\asr-vae\output\bak\asr_vae_ctc_cudnn\v31-2depth-256dim-l21e-4-min1e-2\testbak.csv"
    eval_file = r"D:\Projects\asr-vae\output\bak\asr_vae_ctc_cudnn\v28-2depth-128dim-l21e-4\test.csv"
    eval_col = 2
    evalwords = csv_wordlist(eval_file, eval_col)

    oov = []
    iv = []
    for w in evalwords:
        if w in wordsets[0]:
            iv.append(w)
        else:
            oov.append(w)
    print("Words: {}, OOV: {}, IV: {}".format(len(evalwords), len(oov), len(iv)))
    for w in oov:
        print(w)
