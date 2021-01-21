import tensorflow as tf

from asr_vae.trainer import train, train_flags


def main(_):
    train()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    train_flags(
        # config='conf/wsj_ctc.json',
        # config='conf/wsj_ctc_aae_constlen.json',
        # config='conf/wsj_ctc_vae_constlen.json',
        # config='conf/wsj_ctc_ae_bn.json',
        # config='conf/librispeech_ctc_ae.json',
        config='conf/librispeech_ctc_mm.json',
        # config='conf/wsj_ctc_ae_residual.json',
        # config='conf/wsj_ctc_ae_residual_constlen.json',
        # model_dir='../../output/wsj/ctc-ae/varlen-bn-v2',
        # model_dir='../../output/wsj/ctc-ae/varlen-v3',
        # model_dir='../../output/wsj/ctc-ae/varlen-residual-v4-testtranspose',
        # model_dir='../../output/librispeech/ctc-ae/varlen-nobn-v14',
        model_dir='../../output/librispeech/ctc-mm/ctc-mm-v2',
        train_data_dir='../../data/librispeech/tfrecords/train_clean_360',
        eval_data_dir='../../data/librispeech/tfrecords/dev_clean',
        vocab_file='../../data/librispeech/tfrecords/vocab.npy',
        train_batch_size=4,
        eval_batch_size=4,
        save_summary_steps=100,
        save_summary_steps_slow=400,
        save_checkpoints_steps=2000
    )
    tf.app.run()
