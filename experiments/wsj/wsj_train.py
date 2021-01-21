import tensorflow as tf

from asr_vae.trainer import train, train_flags


def main(_):
    train()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    train_flags(
        config='conf/wsj_ctc.json',
        # config='conf/wsj_ctc_ae_constlen.json',
        #config='conf/wsj_ctc_vae_flat.json',
        #config='conf/wsj_ctc_vae.json',
        # config='conf/wsj_ctc_ae_bn.json',
        # config='conf/wsj_ctc_ae.json',
        # config='conf/wsj_ctc_tf_lstm.json',
        # config='conf/wsj_ctc_variational_residual.json',
        # config='conf/wsj_ctc_ae_residual.json',
        # config='conf/wsj_ctc_ae_residual_constlen.json',
        # model_dir='../../output/wsj/ctc-ae/varlen-bn-v2',
        # model_dir='../../output/wsj/ctc-ae/varlen-v3',
        # model_dir='../../output/wsj/ctc-ae/varlen-residual-v4-testtranspose',
        # model_dir='../../output/wsj/ctc-ae/varlen-latest-v2',
        # model_dir='../../output/wsj/ctc/sentencepieces/lstm-500-5-cnn1d-200-leakyrelu-batchnorm-mm4',
        model_dir='../../output/wsj/acc/wsj-ctc-v2',
        train_data_dir='../../data/wsj/tfrecords-fbank80-logpitch-cmvn-global-sentencepiece-200/train_si284',
        eval_data_dir='../../data/wsj/tfrecords-fbank80-logpitch-cmvn-global-sentencepiece-200/test_dev93',
        data_config='../../data/wsj/tfrecords-fbank80-logpitch-cmvn-global-sentencepiece-200/data_config.json',
        train_batch_size=16,
        eval_batch_size=16,
        save_summary_steps=100,
        save_summary_steps_slow=400,
        # save_checkpoints_steps=2000,
        ctc_mode='sparse',
        allow_growth=True
    )
    tf.app.run()
