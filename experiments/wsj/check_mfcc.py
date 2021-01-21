import os
import re
from collections import Counter

from asr_vae.kaldi.kaldi_records import read_transcripts
from asr_vae.kaldi.kaldi_records import read_ark
d = 'train_si284'
#    'test_dev93',
#    'test_eval92',
#    'test_eval93'
features_path = os.path.join('../../data/wsj/textdata', d,'text-feats.ark')
import matplotlib.pyplot as plt
import numpy as np
def minmax(x):
    print("{}->{}".format(np.min(x), np.max(x)))
for id, features in read_ark(features_path):
    print("Id: {}".format(id))
    print(features.shape)
    plt.imshow(features)
    plt.show()
    minmax(features)
    minmax(features[:,0])
    minmax(features[:,1])
    minmax(features[:,2])
    minmax(features[:,3])
    minmax(features[:,-1])
    minmax(features[:,-2])
    break
