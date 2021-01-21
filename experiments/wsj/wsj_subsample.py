import os

from asr_vae.kaldi.features import extract_features
from asr_vae.kaldi.inputs import dataset_single
import tensorflow as tf
import  numpy as np
def subsample_data(ds:tf.data.Dataset, subsample=3, average=False):
    it = ds.make_initializable_iterator()
    features, labels = it.get_next()
    with tf.train.MonitoredSession() as sess:
        sess.run(it.initializer)
        try:
            while True:
                f,l = sess.run([features, labels])
                f, fl = f
                t, tl = l
                nfl = (-((-fl)//subsample))*subsample
                if nfl > fl:
                    f = np.pad(f, [(0,nfl-fl),(0,0)])
                f = np.reshape(f, (-1, subsample, f.shape[-1]))
                if average:
                    f = np.mean(f, axis=1)
                    yield (f,fl), (t, tl)
                else:
                    for i in range(subsample):
                        fsel = f[:,i,:]
                        yield (fsel,fl), (t, tl)
        except tf.errors.OutOfRangeError:
            print("Done")


if __name__ == '__main__':
    dirs = [
        ['train_si284', False],
        ['test_dev93', True],
        ['test_eval92', True],
        ['test_eval93', True]
    ]
    batch_size = 100
    for d, is_eval in dirs:
        basepath = '../../data/wsj/{}'.format(d)
        records_path = os.path.join(basepath, 'records')
        filenames = tf.data.Dataset.list_files(os.path.join(records_path, '*.tfrecord'), shuffle=False)
        ds = dataset_single(
            filenames=filenames, num_epochs=1, shuffle=False, subsample=1, average=False
        )
        it = ds.make_initializable_iterator()
        features, labels = it.get_next()




        feats_file = os.path.join(basepath, 'feats-extract.ark')
        path = os.path.join(basepath, 'features')
        extract_features(
            feats_file=feats_file,
            path=path)
