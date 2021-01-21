import os
import re

import tensorflow as tf
from tensorflow.python.training import checkpoint_management
from tensorflow_estimator.python.estimator.early_stopping import read_eval_metrics

LEARNING_RATE = 'LEARNING_RATE'


def get_learning_rate():
    collection = tf.get_collection(LEARNING_RATE)
    assert len(collection) == 1
    return collection[0]


def make_learning_rate(lr, eval_metric_ops=None):
    lr = tf.get_variable(
        name="lr",
        dtype=tf.float32,
        shape=[],
        initializer=tf.initializers.constant(lr))
    tf.add_to_collection(
        name=LEARNING_RATE,
        value=lr
    )
    tf.summary.scalar('learning_rate', lr)
    if eval_metric_ops is not None:
        eval_metric_ops['learning_rate'] = lr, tf.no_op()
    return lr


def checkpoint_step(checkpoint_path):
    bn = os.path.basename(checkpoint_path)
    m = re.match("model\\.ckpt-(\\d+)", bn)
    assert m
    return int(m.group(1))


def checkpoint_step_map(checkpoint_paths):
    return {
        checkpoint_step(checkpoint_path): checkpoint_path
        for checkpoint_path in checkpoint_paths
    }


def make_comp_fn_decrease(metric):
    def comp_fn(latest, prev):
        return latest[metric] < prev[metric]

    return comp_fn


def check_train_loop(
        estimator,
        comp_fn,
        epochs_without_improvement,
        input_fn_train,
        input_fn_eval,
        lr_rate,
        lr_min,
        lr_scale
):
    if lr_scale:
        return check_metrics(
            estimator=estimator,
            input_fn_train=input_fn_train,
            input_fn_eval=input_fn_eval,
            comp_fn=comp_fn,
            lr_rate=lr_rate,
            lr_min=lr_min,
            epochs_without_improvement=epochs_without_improvement
        )
    else:
        return check_early_stopping(
            estimator=estimator,
            comp_fn=comp_fn,
            epochs_without_improvement=epochs_without_improvement
        )


def check_early_stopping(estimator, comp_fn, epochs_without_improvement):
    no_improvement, bestk = check_no_improvement(
        estimator=estimator,
        comp_fn=comp_fn,
        epochs_without_improvement=epochs_without_improvement,
        reverse=False
    )
    if no_improvement:
        print("No improvement for {} epochs. Breaking.".format(epochs_without_improvement))
    return no_improvement


def check_no_improvement(estimator, comp_fn, epochs_without_improvement, reverse=False):
    assert epochs_without_improvement >= 1
    metrics = read_eval_metrics(estimator.eval_dir())
    metric_keys = list(metrics.keys())
    metric_keys.sort()

    besti = None
    bestk = None
    best = None
    it = list(enumerate(metric_keys))
    if reverse:
        it.reverse()
    for i, k in it:
        latest = metrics[k]
        if best is None or comp_fn(latest=latest, prev=best):
            besti = i
            bestk = k
            best = latest
    diff = len(metric_keys) - besti - 1
    no_improvement = diff >= epochs_without_improvement
    return no_improvement, bestk


def get_last_step(estimator):
    metrics = read_eval_metrics(estimator.eval_dir())
    metric_keys = list(metrics.keys())
    metric_keys.sort()
    return metric_keys[-1]


def check_metrics(estimator, input_fn_train, input_fn_eval, comp_fn, lr_rate, lr_min, epochs_without_improvement):
    no_improvement, bestk = check_no_improvement(
        estimator=estimator,
        comp_fn=comp_fn,
        epochs_without_improvement=epochs_without_improvement,
        reverse=False
    )
    if no_improvement:
        no_improvement, bestk = check_no_improvement(
            estimator=estimator,
            comp_fn=comp_fn,
            epochs_without_improvement=epochs_without_improvement,
            reverse=True
        )
        checkpoints = checkpoint_management.get_checkpoint_state(estimator.model_dir)
        checkpoints = checkpoint_step_map(checkpoints.all_model_checkpoint_paths)
        prev_checkpoint = checkpoints[bestk]
        with tf.Graph().as_default():
            features, labels, input_hooks = (
                estimator._get_features_and_labels_from_input_fn(
                    input_fn_train, tf.estimator.ModeKeys.TRAIN))
            estimator_spec = estimator._call_model_fn(
                features, labels, tf.estimator.ModeKeys.TRAIN, estimator.config)
            lr = get_learning_rate()
            lr_up = tf.assign(lr, lr * lr_rate)
            step = tf.train.get_or_create_global_step()
            last_step = get_last_step(estimator)
            step_up = tf.assign(step, last_step + 1)
            with tf.train.MonitoredSession(
                    session_creator=tf.train.ChiefSessionCreator(
                        checkpoint_filename_with_path=prev_checkpoint,
                        scaffold=estimator_spec.scaffold,
                        config=estimator._session_config)) as session:
                lr_new, step_new = session.run([lr_up, step_up])
                savers = tf.get_collection(tf.GraphKeys.SAVERS)
                assert len(savers) == 1
                saver = savers[0]

                def step_fn(step_context):
                    saver.save(step_context.session,
                               os.path.join(
                                   estimator.model_dir,
                                   "model.ckpt-{}".format(step_new)
                               ))

                session.run_step_fn(step_fn)

        estimator.evaluate(
            input_fn=input_fn_eval
        )
        if lr_new < lr_min:
            print("Minimum learning rate reached. Breaking.")
            return True
    return False


class SimpleSaver(tf.train.CheckpointSaverHook):
    def __init__(
            self,
            checkpoint_dir,
            saver=None,
            checkpoint_basename="model.ckpt",
            scaffold=None,
            listeners=None):
        super(SimpleSaver, self).__init__(
            checkpoint_dir=checkpoint_dir,
            save_secs=None,
            save_steps=1,
            saver=saver,
            checkpoint_basename=checkpoint_basename,
            scaffold=scaffold,
            listeners=listeners)

    def before_run(self, run_context):  # pylint: disable=unused-argument
        pass

    def after_run(self, run_context, run_values):
        pass

    def end(self, session):
        last_step = session.run(self._global_step_tensor)
        self._save(session, last_step)
        for l in self._listeners:
            l.end(session, last_step)
