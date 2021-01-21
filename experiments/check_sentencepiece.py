import sentencepiece as spm
import tensorflow as tf

from asr_vae.kaldi.inputs import sentencepiece_op_func

model_path = r"D:\Projects\asr-vae\data\wsj\tfrecords-fbank80-logpitch-cmvn-global-sentencepiece-100\sentencepiece-model.model"
sp = spm.SentencePieceProcessor()
sp.LoadOrDie(model_path)
x = tf.constant("HE SAYS ITS BENEFICIARIES ARE CHARITIES WHICH HE PROMISES TO IDENTIFY SOMEDAY SOON")
y = sentencepiece_op_func(inp=x, sp=sp, alpha=0.2)
print("X")
print(x)
print("Y")
print(y)

with tf.train.MonitoredSession() as sess:
    for i in range(10):
        ids = sess.run(y)
        #print(ids)
        ids = [int(j) for j in ids]
        #print(ids)
        tokens = [sp.IdToPiece(j) for j in ids]
        print(tokens)
        #s = sp.DecodeIds(list(ids))
        #print(s)
