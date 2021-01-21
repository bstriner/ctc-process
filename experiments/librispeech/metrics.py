import csv

import edit_distance
import numpy as np
import tensorflow as tf
from jiwer import wer


def main(argv):
    errs = []
    wcs = []
    eds = []
    ccs = []
    correct = 0
    total = 0
    with open(tf.flags.FLAGS.generated, 'r') as f:
        r = csv.reader(f)
        for i, row in enumerate(r):
            if i > 0:
                truth = row[1]
                hypothesis = row[2]
                err = wer(truth=truth, hypothesis=hypothesis)
                wc = len(truth.split(" "))
                errs.append(err*wc)
                wcs.append(wc)
                ed = edit_distance.edit_distance(truth, hypothesis)[0]
                cc = len(truth)
                eds.append(ed)
                ccs.append(cc)
                total += 1
                if truth==hypothesis:
                    correct+=1
    print(np.sum(errs) / np.sum(wcs))
    print(np.mean(np.divide(errs, wcs)))
    print(np.sum(eds) / np.sum(ccs))
    print(correct/total)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    #tf.flags.DEFINE_string('generated', '../../output/librispeech/lstm/ctc_vae/v1/testbak.csv', 'Model directory')
    tf.flags.DEFINE_string('generated', '../../output/asr_ctc_cudnn/v24-half-l21e-4-depth2-mean-dim256/test.csv', 'Model directory')

    tf.app.run()

"""
VAE
0.19183054232601013
0.2064042721630605
0.06529271249542692
0.12654320987654322

CTC
0.25480104470732834
0.27105103001140485
0.08259506610331198
0.07407407407407407
"""