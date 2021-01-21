import os

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

"""
def train(rundir, hparams):
    run_config = RunConfig(
        model_dir=rundir
    )
    with tf.summary.FileWriter(rundir) as w:
        w.add_summary(hparams_config)
        w.add_summary(summary_scalar_pb, global_step=1)
    # with tf.summary.FileWriter(run_config.model_dir).as_default():
    with tf.contrib.summary.create_file_writer(os.path.join(run_config.model_dir, 'hparams')).as_default():
        hp.hparams(hparams.values())
    # Train Data
    train_input_fn = make_input_fn(
        train_data_dir,
        batch_size=train_batch_size,
        shuffle=True,
        num_epochs=None,
        subsample=hparams.subsample,
        average=False)

    # Test Data
    eval_input_fn = make_input_fn(
        eval_data_dir,
        batch_size=eval_batch_size,
        shuffle=False,
        num_epochs=1,
        subsample=hparams.subsample,
        average=True)

    # Vocab
    vocab = np.load(vocab_file)

    # Model
    model_fn = make_model_fn(hparams=hparams, run_config=run_config, vocab=vocab)
    estimator = Estimator(
        model_fn=model_fn,
        config=run_config,
        params=hparams)
    early_stopping_hook = stop_if_no_decrease_hook(
        estimator=estimator,
        #metric_name="loss",
        metric_name="character_error_rate",
        max_steps_without_decrease=max_steps_without_decrease,
        min_steps=min_steps)
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=max_steps,
        hooks=[early_stopping_hook])
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        steps=eval_steps,
        throttle_secs=0)
    train_and_evaluate(
        eval_spec=eval_spec,
        train_spec=train_spec,
        estimator=estimator
    )

def train_test_model(hparams):
    pass
"""

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 64]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
HP_MODE = hp.HParam('mode', hp.Discrete(['train', 'eval']))

METRIC_ACCURACY = 'accuracy'
logdir = 'hparam_tuning'
# os.makedirs(logdir, exist_ok=True)

if False:
    hparams_config = hp.hparams_config_pb(
        hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER, HP_MODE],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')]
    ).SerializeToString()
    with tf.summary.FileWriter(logdir) as w:
        w.add_summary(hparams_config)


session_num = 1
for num_units in HP_NUM_UNITS.domain.values:
    for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
        for optimizer in HP_OPTIMIZER.domain.values:
            tf.reset_default_graph()
            hparams = {
                HP_NUM_UNITS.name: num_units,
                HP_DROPOUT.name: dropout_rate,
                HP_OPTIMIZER.name: optimizer,
                HP_MODE.name: 'train'
            }
            hparams_eval = {
                HP_NUM_UNITS.name: num_units,
                HP_DROPOUT.name: dropout_rate,
                HP_OPTIMIZER.name: optimizer,
                HP_MODE.name: 'eval'
            }
            rundir = os.path.join(logdir, "{}-{}-{}".format(num_units, dropout_rate, optimizer))
            # rundir = os.path.join(logdir, "{}".format(session_num))
            evaldir = os.path.join(rundir, 'eval')
            hparams_pb = hp.hparams_pb(hparams).SerializeToString()  # record the values used in this trial
            hparams_pb_eval = hp.hparams_pb(hparams_eval).SerializeToString()  # record the values used in this trial
            accuracy = 0.9
            eval_accuracy = 0.8
            summary_scalar = tf.summary.scalar(METRIC_ACCURACY, tf.random.uniform(shape=[]))
            with tf.train.MonitoredSession() as sess:
                summary_scalar_pb = sess.run(summary_scalar)
                summary_scalar_eval_pb = sess.run(summary_scalar)
            with tf.summary.FileWriter(rundir) as w:
                w.add_summary(hparams_pb)
                w.add_summary(summary_scalar_pb, global_step=1)
            with tf.summary.FileWriter(evaldir) as w:
                w.add_summary(hparams_pb_eval)
                w.add_summary(summary_scalar_eval_pb, global_step=1)
            session_num += 1
