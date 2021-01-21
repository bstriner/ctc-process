import tensorflow as tf

from asr_vae.evaluator import evaluate


def main(_):
    evaluate()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('model_dir', '../../output/librispeech/ctc/v1', 'Model directory')
    tf.flags.DEFINE_string('prediction_path', '../../output/librispeech/ctc/v1/evaluation.csv', 'Data directory')
    tf.flags.DEFINE_string('eval_data_dir', '../../data/librispeech/test_clean/records', 'Data directory')
    tf.flags.DEFINE_string('vocab_file', '../../data/librispeech/vocab.npy', 'Data directory')
    tf.flags.DEFINE_integer('train_batch_size', 32, 'Batch size')
    tf.flags.DEFINE_integer('eval_batch_size', 32, 'Batch size')
    tf.flags.DEFINE_integer('save_checkpoints_steps', 1000, 'save_checkpoints_steps')
    tf.flags.DEFINE_integer('max_steps', 80000, 'max_steps')
    tf.flags.DEFINE_integer('buffer_size', 1000, 'capacity')
    tf.flags.DEFINE_bool('debug', default=False, help='debugging')
    tf.app.run()
