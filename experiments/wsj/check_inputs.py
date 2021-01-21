import os

import tensorflow as tf
import matplotlib.pyplot as plt
from asr_vae.kaldi.inputs import dataset_batch, dataset_single
import glob
import numpy as np
if __name__ == '__main__':
    path = '../../data/wsj/test_eval92/records'
    fpath = '../../data/wsj/test_eval92/features'
    tpath = '../../data/wsj/test_eval92/transcripts'

    ffiles = list(glob.glob(os.path.join(fpath, '*.npy')))
    ffiles.sort()
    tfiles = list(glob.glob(os.path.join(tpath, '*.npy')))
    tfiles.sort()

    files = tf.data.Dataset.list_files(os.path.join(path, '*.tfrecord'), shuffle=False)
    ds_single = dataset_single(files, shuffle=False, num_epochs=1)
    batch_size = 5
    ds = dataset_batch(ds_single, batch_size=batch_size)
    it = ds.make_initializable_iterator()
    (f, fl), (t, tl) = it.get_next()
    with tf.train.MonitoredSession() as sess:
        # sess.run(tf.global_variables_initializer())
        sess.run(it.initializer)
        # sess.run(tf.train.start_queue_runners())
        _f, _fl, _t, _tl = sess.run([f, fl, t, tl])
        print(_tl)
        print(_t.shape)
        print(_fl)
        print(_f.shape)

        for i in range(batch_size):
            mel = _f[i, :_fl[i], :]
            plt.figure()
            plt.imshow(mel)
            plt.show()

            m2 = np.load(ffiles[i])
            plt.figure()
            plt.imshow(m2)
            plt.show()



        # for i in range(5):
        #    _x = sess.run(x)
        #    print(_x)
        # _f, _fl, _t, _tl = sess.run(x)
        # print(_fl)
        # print(_t)
