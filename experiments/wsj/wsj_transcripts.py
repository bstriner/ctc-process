import os

from asr_vae.kaldi.transcripts import extract_transcripts

if __name__ == '__main__':
    dirs = [
        'train_si284',
        'test_dev93',
        'test_eval92',
        'test_eval93'
    ]
    vocab_file = '../../data/wsj/vocab.npy'
    for d in dirs:
        basepath = '../../data/wsj/{}'.format(d)
        text_file = os.path.join(basepath, 'text')
        path = os.path.join(basepath, 'transcripts')
        extract_transcripts(
            text_file=text_file,
            vocab_file=vocab_file,
            path=path)
