import os

from asr_vae.kaldi.features import extract_features

if __name__ == '__main__':
    dirs = [
        'train_si284',
        'test_dev93',
        'test_eval92',
        'test_eval93'
    ]
    for d in dirs:
        basepath = '../../data/wsj/{}'.format(d)
        feats_file = os.path.join(basepath, 'feats-extract.ark')
        path = os.path.join(basepath, 'features')
        extract_features(
            feats_file=feats_file,
            path=path)
