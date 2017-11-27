import tensorflow as tf
from tensorflow.python.framework import dtypes

def tf_hash_table(keys, values):
    table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(keys, values, tf.int64, tf.int64), -1)
    return table



net = tf.Variable([ range(0, 25), range(0, 25) , range(0, 25), range(0, 25)], tf.float32)


values_lower_padding = [16, 8, 23, 12, 0, 7, 19]
keys = [0, 1, 2, 3, 4, 5, 6]
basenet = tf.constant([ [0.7, 0.3, 0, 0], [0.2, 30, 2, 0] , [0.1, 0.7, 0.2, 0], [0.1, 0.7, 0.2, 0] ])
basenet_key = tf.argmax(basenet, 1)

lower_table = tf_hash_table(keys, values_lower_padding)
lower_value = lower_table.lookup(basenet_key)
lower_value = tf.cast(lower_value, tf.int32)


# keys = [0, 1, 2, 3, 4, 5, 6]
# values_upper_padding = [6, 13, 0, 9, 18, 17, 2]
# values_lower_padding = [16, 8, 23, 12, 0, 7, 19]
sess = tf.Session()
# keys = tf.constant(keys, tf.int64)
#values = tf.constant(values, tf.int64)

      # a hash table is defined for translating
# table = tf.contrib.lookup.HashTable(
# tf.contrib.lookup.KeyValueTensorInitializer(keys, values_upper_padding, tf.int64, tf.int64), -1
#       )

lower_table.init.run(session = sess)
init = tf.initialize_all_variables()
sess.run(init)

# print sess.run(net)
# print sess.run(lower_value[0])
# print sess.run(tf.zeros([lower_value[0]], tf.float32))
# print sess.run(net[0, lower_value[0]:])

for i in range(0, lower_value.shape[0]):
    padding = tf.zeros([lower_value[i]], tf.int32)
    cropped_net = net[i, lower_value[i]:]
    #print sess.run([padding, cropped_net])
    tmp = tf.concat([padding, cropped_net], 0)
    k = tf.assign(net[i], tmp)
#    tmp = tf.concat([tf.zeros([lower_value[i]], tf.float32) ,net[i, lower_value[i]:]], 0)
    sess.run(k)

print sess.run(k)




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