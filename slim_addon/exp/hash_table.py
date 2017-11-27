import tensorflow as tf
from tensorflow.python.framework import dtypes


keys = [1, 2, 3]
values = [10, 20, 30]
sess = tf.Session()

label = tf.constant(1, tf.int64)

keys = tf.constant(keys, dtypes.int64)
values = tf.constant(values, dtypes.int64)

      # a hash table is defined for translating
table = tf.contrib.lookup.HashTable(
tf.contrib.lookup.KeyValueTensorInitializer(keys, values, dtypes.int64, dtypes.int64), -1
      )
table.init.run(session = sess)
out = table.lookup(label)
print(sess.run(out))