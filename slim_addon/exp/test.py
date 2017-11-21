import tensorflow as tf
from tensorflow.python.framework import dtypes

def tf_hash_table(keys, values):
    table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(keys, values, tf.int64, tf.int64), -1)
    return table

keys = [0, 1, 2, 3, 4, 5, 6]
values_upper_padding = [6, 13, 0, 9, 18, 17, 2]
values_lower_padding = [16, 8, 23, 12, 0, 7, 19]
sess = tf.Session()
keys = tf.constant(keys, tf.int64)
#values = tf.constant(values, tf.int64)

      # a hash table is defined for translating
# table = tf.contrib.lookup.HashTable(
# tf.contrib.lookup.KeyValueTensorInitializer(keys, values_upper_padding, tf.int64, tf.int64), -1
#       )

table = tf_hash_table(keys, values_upper_padding)
table.init.run(session = sess)

label = tf.constant([0, 1, 0, 0, 0, 0.5])
label = tf.argmax(label)
#label = tf.cast(label, tf.int32)

out = table.lookup(label)
print tf.rank(tf.constant([[1, 2, 3], [2, 3, 4]]))
print tf.rank(label)
print tf.rank(out)

print(sess.run([label, tf.rank(out), tf.reshape(out, [1, 1])]))



    # keys = [0, 1, 2, 3, 4, 5, 6]

    # values_upper_padding = [6, 13, 0, 9, 18, 17, 2]
    # values_lower_padding = [16, 8, 23, 12, 0, 7, 19]

    # upper_table = tf_hash_table(keys, values_upper_padding)
    # lower_table = tf_hash_table(keys, values_lower_padding)
    # basenet_key = tf.argmax(basenet)
    # basenet_key = tf.cast(basenet_key, tf.int64)
    # upper_value = upper_table.lookup(basenet_key)
    # lower_value = lower_table.lookup(basenet_key)

    # x = tf.reshape(upper_value, [1,1])
    # y = tf.reshape(lower_value, [1,1])