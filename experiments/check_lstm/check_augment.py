import imageio
import tensorflow as tf

from asr_vae.kaldi.spec_augment import SpecAugmentParams, augment

sa = SpecAugmentParams(
    W=80,
    F=30,
    T=40,
    mF=2,
    mT=2,
    p=1.,
    enabled=True
)

tl = 200
dim = 80
# x = tf.random.normal(
#    shape=(tl, dim)
# )
mel = tf.placeholder(
    name='x',
    shape=(None, None),
    dtype=tf.float32
)
xsa = augment(
    utterance=mel,
    sa_params=sa
)
real_mel = imageio.imread('mel.png')
print(mel)
print(xsa)
with tf.train.MonitoredSession() as sess:
    x1, x2 = sess.run([mel, xsa], feed_dict={
        mel: real_mel
    })
    imageio.imwrite('mel-original.png', x1)
    imageio.imwrite('mel-augmented.png', x2)
