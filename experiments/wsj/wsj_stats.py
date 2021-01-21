import csv
import os

import numpy as np
import tqdm

from asr_vae.kaldi.records import check_file_list

if __name__ == '__main__':
    transcripts_path = '../../data/wsj/train_si284/transcripts'
    features_path = '../../data/wsj/train_si284/features'
    files = list(check_file_list(transcripts_path, features_path))
    with open('../../output/wsj/data_stats.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Id', 'Utterance Length', 'Transcript Length', 'Ratio'])
        for i, file in tqdm.tqdm(enumerate(files)):
            ut = np.load(os.path.join(features_path, file))
            tr = np.load(os.path.join(transcripts_path, file))
            ul = ut.shape[0]
            tl = tr.shape[0]
            w.writerow([i, ul, tl, float(ul) / float(tl)])
