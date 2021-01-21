import tensorflow as tf

from asr_vae.evaluator import evaluate


def main(_):
    evaluate()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    #tf.flags.DEFINE_string('model_dir', '../../output/wsj/wsj-latest-1e3/wsj-ctc-fbank80-logpitch-cmvn-global-sentencepiece-200-cudnn_lstm-500-5-adamw-1e-3-1e-6-clipping-0.1-sub-3', 'Model directory')
    #tf.flags.DEFINE_string('prediction_path', '../../output/wsj/wsj-latest-1e3/wsj-ctc-fbank80-logpitch-cmvn-global-sentencepiece-200-cudnn_lstm-500-5-adamw-1e-3-1e-6-clipping-0.1-sub-3/evaluation.csv', 'Data directory')
    tf.flags.DEFINE_string('model_dir',
                           '../../output/wsj/wsj-latest-1e3/wsj-ctc-mm-fbank80-logpitch-cmvn-global-sentencepiece-200-cudnn_lstm-500-5-adamw-1e-3-1e-6-clipping-0.1-sub-3-mm-4',
                           'Model directory')
    tf.flags.DEFINE_string('prediction_path',
                           '../../output/wsj/wsj-latest-1e3/wsj-ctc-mm-fbank80-logpitch-cmvn-global-sentencepiece-200-cudnn_lstm-500-5-adamw-1e-3-1e-6-clipping-0.1-sub-3-mm-4/evaluation.csv',
                           'Data directory')
    #tf.flags.DEFINE_string('model_dir',
    #                   '../../output/wsj/wsj-latest-1e3/wsj-vae-fbank80-logpitch-cmvn-global-sentencepiece-200-cudnn_lstm-500-5-adamw-1e-3-1e-6-clipping-0.1-sub-3-encoder-320-3-flatlatent',
    #                   'Model directory')
    #tf.flags.DEFINE_string('prediction_path',
    #                   '../../output/wsj/wsj-latest-1e3/wsj-vae-fbank80-logpitch-cmvn-global-sentencepiece-200-cudnn_lstm-500-5-adamw-1e-3-1e-6-clipping-0.1-sub-3-encoder-320-3-flatlatent/evaluation.csv',
    #                   'Data directory')
    #tf.flags.DEFINE_string('model_dir',
    #                       '../../output/wsj/wsj-latest-1e3/wsj-vae-fbank80-logpitch-cmvn-global-sentencepiece-200-cudnn_lstm-500-5-adamw-1e-3-1e-6-clipping-0.1-sub-3-encoder-320-3-varlatent',
    #                       'Model directory')
    #tf.flags.DEFINE_string('prediction_path',
    #                       '../../output/wsj/wsj-latest-1e3/wsj-vae-fbank80-logpitch-cmvn-global-sentencepiece-200-cudnn_lstm-500-5-adamw-1e-3-1e-6-clipping-0.1-sub-3-encoder-320-3-varlatent/evaluation.csv',
    #                       'Data directory')

    tf.flags.DEFINE_string('eval_data_dir', '../../data/wsj/tfrecords-fbank80-logpitch-cmvn-global-sentencepiece-200/test_eval92', 'Data directory')
    tf.flags.DEFINE_string('data_config', '../../data/wsj/tfrecords-fbank80-logpitch-cmvn-global-sentencepiece-200/data_config.json', 'Data directory')
    tf.flags.DEFINE_integer('train_batch_size', 16, 'Batch size')
    tf.flags.DEFINE_integer('eval_batch_size', 16, 'Batch size')
    tf.flags.DEFINE_integer('save_checkpoints_steps', 1000, 'save_checkpoints_steps')
    tf.flags.DEFINE_integer('max_steps', 80000, 'max_steps')
    tf.flags.DEFINE_integer('buffer_size', 1000, 'capacity')
    tf.app.flags.DEFINE_integer('num_parallel_calls', 4, 'num_parallel_calls')
    tf.app.flags.DEFINE_integer('prefetch_buffer_size', 100, 'prefetch_buffer_size')
    tf.app.flags.DEFINE_string('ctc_mode', 'sparse', 'ctc_mode')
    tf.app.flags.DEFINE_integer('beam_width', 1, 'beam_width')
    tf.app.flags.DEFINE_integer('save_summary_steps', 100, 'max_steps')
    tf.app.flags.DEFINE_integer('save_summary_steps_slow', 100, 'max_steps')
    tf.app.flags.DEFINE_integer('random_search_iter', 256, 'random_search')
    tf.app.flags.DEFINE_boolean('random_search', True, 'random_search')
    tf.flags.DEFINE_bool('debug', default=False, help='debugging')
    tf.app.run()


"""
ctc
Char rate: 0.06360795625262905
Word rate: 0.19298245614035087

mm-4
Char rate: 0.0687759149089598
Word rate: 0.21194400141768563

vae-flatlatent
Char rate: 0.06751397151613485
Word rate: 0.20077972709551656

vae-varlatent
Char rate: 0.06736374015984617
Word rate: 0.20839978734715578

"""