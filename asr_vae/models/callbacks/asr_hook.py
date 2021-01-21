import csv
import os

import tensorflow as tf
from tensorflow.python.training.session_run_hook import SessionRunHook


class AsrHook(SessionRunHook):
    def __init__(self, true_strings, generated_strings, path):
        self.true_strings = true_strings
        self.generated_strings = generated_strings
        self.path = path
        self.step = tf.train.get_or_create_global_step()

    def after_create_session(self, session, coord):
        print("AsrHook Running hook")
        true, gen, step = session.run([self.true_strings, self.generated_strings, self.step])
        print("AsrHook Ran hook")
        n = true.shape[0]
        assert n == gen.shape[0]
        assert true.ndim == 1
        assert gen.ndim == 1
        target_path = self.path.format(step)
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        with open(target_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Row', 'True', 'Generated'])
            for i in range(n):
                w.writerow([i, true[i].decode(), gen[i].decode()])
