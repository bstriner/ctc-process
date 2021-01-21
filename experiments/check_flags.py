import tensorflow as tf

tf.flags.DEFINE_bool("mybool", True, 'help text')

print("mybool" in tf.flags.FLAGS)
print("mybool2" in tf.flags.FLAGS)

print(tf.flags.FLAGS.mybool)
print()

print(tf.flags.FLAGS.mybool2)