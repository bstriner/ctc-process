import tensorflow as tf
from tensorflow.python.training.session_run_hook import SessionRunHook


class GradientAccumulatorHook(SessionRunHook):
    def __init__(self, loss, var_list, opt: tf.train.Optimizer, global_step, frequency=5, transform_grads_fn=None):
        grads_and_vars = opt.compute_gradients(loss=loss, var_list=var_list)
        accumulators = [
            tf.Variable(shape=v.shape, dtype=tf.identity(v).dtype, initial_value=tf.zeros_like(v),
                        trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
            for g, v in grads_and_vars
        ]
        self.global_step = global_step
        self.frequency = frequency

        # Train op
        updates = [
            tf.assign_add(acc, g)
            for acc, (g, v) in
            zip(accumulators, grads_and_vars)]
        self.counter = tf.Variable(shape=[], dtype=tf.int32, initial_value=tf.zeros(shape=[], dtype=tf.int32),
                                   trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
        with tf.control_dependencies(updates):
            with tf.control_dependencies([tf.assign_add(global_step, 1)]):
                self.train_op = tf.identity(loss)

        # Apply op
        grads_and_vars_acc = [
            (acc, v)
            for acc, (g, v) in
            zip(accumulators, grads_and_vars)]
        if transform_grads_fn:
            grads_and_vars_acc = transform_grads_fn(grads_and_vars_acc)
        apply_gradients = opt.apply_gradients(grads_and_vars=grads_and_vars_acc, global_step=None)
        with tf.control_dependencies([apply_gradients]):
            zeros = [
                tf.assign(acc, tf.zeros_like(acc))
                for acc in accumulators
            ]
            self.apply_op = tf.group(zeros)

    def before_run(self, run_context: tf.estimator.SessionRunContext):
        return tf.estimator.SessionRunArgs(fetches=[self.global_step])

    def after_run(self,
                  run_context: tf.estimator.SessionRunContext,
                  run_values: tf.estimator.SessionRunValues):
        counter = run_values.results[0]
        print("Counter: {}".format(counter))
        if counter % self.frequency == 0:
            run_context.session.run(self.apply_op)
