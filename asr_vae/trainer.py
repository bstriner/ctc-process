import json
import os

import numpy as np
import sentencepiece as spm
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.run_config import RunConfig

from .default_params import get_hparams
from .kaldi.inputs import make_input_fn
from .kaldi.spec_augment import SpecAugmentParams
from .models.model import make_model_fn
from .train_loop import SimpleSaver, check_train_loop, make_comp_fn_decrease


def train():
    model_dir = tf.app.flags.FLAGS.model_dir
    # max_steps_without_decrease = tf.app.flags.FLAGS.max_steps_without_decrease
    # min_steps = tf.app.flags.FLAGS.min_steps
    # save_checkpoints_steps = tf.app.flags.FLAGS.save_checkpoints_steps
    train_data_dir = tf.app.flags.FLAGS.train_data_dir
    train_batch_size = tf.app.flags.FLAGS.train_batch_size
    eval_data_dir = tf.app.flags.FLAGS.eval_data_dir
    eval_batch_size = tf.app.flags.FLAGS.eval_batch_size
    # max_steps = tf.app.flags.FLAGS.max_steps
    # eval_steps = tf.app.flags.FLAGS.eval_steps
    # save_summary_steps = tf.app.flags.FLAGS.save_summary_steps
    distribute = tf.app.flags.FLAGS.distribute
    os.makedirs(model_dir, exist_ok=True)
    print("model_dir={}".format(model_dir))
    if distribute:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = None
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = tf.app.flags.FLAGS.allow_growth
    hparams = get_hparams(model_dir, validate=True)
    hparams_dict = hparams.values()
    hparams_dict_train = hparams_dict.copy()
    hparams_dict_eval = hparams_dict.copy()
    hparams_dict_train['mode'] = 'train'
    hparams_dict_eval['mode'] = 'eval'
    hparams_pb = hp.hparams_pb(hparams_dict_train).SerializeToString()
    hparams_pb_eval = hp.hparams_pb(hparams_dict_eval).SerializeToString()
    with tf.summary.FileWriter(model_dir) as w:
        w.add_summary(hparams_pb)
    with tf.summary.FileWriter(os.path.join(model_dir, 'eval')) as w:
        w.add_summary(hparams_pb_eval)

    run_config = RunConfig(
        model_dir=model_dir,
        train_distribute=strategy,
        eval_distribute=strategy,
        save_checkpoints_steps=None,
        save_summary_steps=None,
        session_config=session_config,
        keep_checkpoint_max=max(hparams.epochs_without_improvement + 1, 5)
    )

    # Vocab
    with open(tf.app.flags.FLAGS.data_config, 'r') as fp:
        data_config = json.load(fp)
        input_dim = data_config['input_dim']
        vocab = data_config['vocab']
        vocab = np.array(vocab, dtype=np.unicode)
        sentencepiece = data_config['sentencepiece']
    if hparams.sentencepiece_online:
        spmodel = os.path.join(
            os.path.dirname(os.path.abspath(tf.app.flags.FLAGS.data_config)),
            "sentencepiece-model.model")
        sp = spm.SentencePieceProcessor()
        sp.LoadOrDie(spmodel)
    else:
        sp = None
    sa_params = SpecAugmentParams.from_params(hparams)
    # Train Data
    train_input_fn = make_input_fn(
        train_data_dir,
        batch_size=train_batch_size,
        shuffle=True,
        num_epochs=1,
        subsample=hparams.subsample,
        independent_subsample=hparams.independent_subsample,
        average=False,
        bucket=hparams.bucket,
        bucket_size=hparams.bucket_size,
        buckets=hparams.buckets,
        sa_params=sa_params,
        input_dim=input_dim,
        sp=sp
    )

    # Test Data
    eval_input_fn = make_input_fn(
        eval_data_dir,
        batch_size=eval_batch_size,
        shuffle=False,
        num_epochs=1,
        subsample=hparams.subsample,
        independent_subsample=hparams.independent_subsample,
        average=True,
        bucket=hparams.bucket,
        bucket_size=hparams.bucket_size,
        buckets=hparams.buckets,
        sa_params=None,
        input_dim=input_dim
    )

    # Model
    model_fn = make_model_fn(run_config=run_config, vocab=vocab, sentencepiece=sentencepiece)
    estimator = Estimator(
        model_fn=model_fn,
        config=run_config,
        params=hparams)
    print("estimator.eval_dir(): {}".format(estimator.eval_dir()))
    if hparams.model == 'ae':
        metric_name = 'autoencoded_character_error_rate'
    else:
        metric_name = "character_error_rate"
    saver_hook = SimpleSaver(estimator.model_dir)
    comp_fn = make_comp_fn_decrease(metric=metric_name)
    estimator.evaluate(
        input_fn=eval_input_fn
    )
    while True:
        estimator.train(
            input_fn=train_input_fn,
            hooks=[saver_hook]
        )
        estimator.evaluate(
            input_fn=eval_input_fn
        )
        if check_train_loop(
                epochs_without_improvement=hparams.epochs_without_improvement,
                lr_scale=hparams.lr_scale,
                estimator=estimator,
                input_fn_train=train_input_fn,
                input_fn_eval=eval_input_fn,
                comp_fn=comp_fn,
                lr_rate=hparams.lr_rate,
                lr_min=hparams.lr_min):
            break


def train_flags(
        config,
        model_dir,
        train_data_dir,
        eval_data_dir,
        # vocab_file,
        data_config,
        train_batch_size,
        eval_batch_size,
        ctc_mode='sparse',
        # save_checkpoints_steps=2000,
        save_summary_steps=100,
        save_summary_steps_slow=400,
        # max_steps=200000,
        allow_growth=False,
        train_batch_acc=1,
        beam_width=100):
    tf.app.flags.DEFINE_string('config', config, 'config file')
    tf.app.flags.DEFINE_string('data_config', data_config, 'data_config file')
    tf.app.flags.DEFINE_string('hparams', '', 'hparam keys/values')
    tf.app.flags.DEFINE_string('model_dir', model_dir, 'Model directory')
    tf.app.flags.DEFINE_string('train_data_dir', train_data_dir, 'Data directory')
    tf.app.flags.DEFINE_string('eval_data_dir', eval_data_dir, 'Data directory')
    # tf.app.flags.DEFINE_string('vocab_file', vocab_file, 'Data directory')
    tf.app.flags.DEFINE_string('ctc_mode', ctc_mode, 'ctc_mode')
    tf.app.flags.DEFINE_integer('train_batch_size', train_batch_size, 'Batch size')
    tf.app.flags.DEFINE_integer('eval_batch_size', eval_batch_size, 'Batch size')
    # tf.app.flags.DEFINE_integer('save_checkpoints_steps', save_checkpoints_steps, 'save_checkpoints_steps')
    # tf.app.flags.DEFINE_integer('max_steps', max_steps, 'max_steps')
    # tf.app.flags.DEFINE_integer('min_steps', 50000, 'Batch size')
    tf.app.flags.DEFINE_integer('beam_width', beam_width, 'beam_width')
    # tf.app.flags.DEFINE_integer('max_steps_without_decrease', 50000, 'Batch size')
    # tf.app.flags.DEFINE_integer('eval_steps', 200, 'max_steps')
    tf.app.flags.DEFINE_integer('save_summary_steps', save_summary_steps, 'max_steps')
    tf.app.flags.DEFINE_integer('save_summary_steps_slow', save_summary_steps_slow, 'max_steps')
    tf.app.flags.DEFINE_integer('shuffle_buffer_size', 1000, 'shuffle_buffer_size')
    tf.app.flags.DEFINE_integer('prefetch_buffer_size', 100, 'prefetch_buffer_size')
    tf.app.flags.DEFINE_integer('num_parallel_calls', 4, 'num_parallel_calls')
    tf.app.flags.DEFINE_bool('debug', default=False, help='debugging')
    tf.app.flags.DEFINE_bool('distribute', default=False, help='distribute')
    tf.app.flags.DEFINE_bool('allow_growth', default=allow_growth, help='allow_growth')
