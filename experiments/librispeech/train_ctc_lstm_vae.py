import tensorflow as tf

import asr_vae.trainer
from asr_vae.models.ctc_vae_model import make_ctc_vae_model_fn


def main(argv):
    asr_vae.trainer.main(argv, make_ctc_vae_model_fn)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('model_dir', '../../output/librispeech/lstm/ctc_vae/v3', 'Model directory')
    tf.flags.DEFINE_string('data_dir', '../../dataset', 'Data directory')
    tf.flags.DEFINE_string('schedule', 'train_and_evaluate', 'Schedule')
    # tf.flags.DEFINE_string('schedule', 'test', 'Schedule')
    tf.flags.DEFINE_integer('batch_size', 16, 'Batch size')
    tf.flags.DEFINE_integer('capacity', 4000, 'capacity')
    tf.flags.DEFINE_integer('max_steps', 100000, 'max_steps')
    tf.flags.DEFINE_integer('min_after_dequeue', 2000, 'min_after_dequeue')
    tf.flags.DEFINE_integer('grid_size', 10, 'grid_size')
    tf.flags.DEFINE_integer('queue_threads', 2, 'queue_threads')
    tf.flags.DEFINE_integer('save_checkpoints_steps', 2000, 'save_checkpoints_secs')
    tf.flags.DEFINE_string('hparams', '', 'Hyperparameters')
    tf.app.run()
