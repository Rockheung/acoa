import tensorflow as tf

def tf_hash_table(keys, values):
    table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(keys, values, tf.int64, tf.int64), -1)
    return table

keys = [0, 1, 2, 3, 4, 5, 6]
values_lower_padding = [16, 8, 23, 12, 0, 7, 19]

basenet = tf.constant([ [0.7, 0.3, 0, 0], [0.2, 30, 2, 0] , [0.1, 0.7, 0.2, 0] ])
basenet_key = tf.argmax(basenet, 1)

lower_table = tf_hash_table(keys, values_lower_padding)
lower_value = lower_table.lookup(basenet_key)
lower_value = tf.cast(lower_value, tf.int32)


with tf.Session() as sess:
	lower_table.init.run(session = sess)
	print(sess.run([lower_value, tf.rank(lower_value)]))
