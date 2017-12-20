import tensorflow as tf
#########################ACOA###############################
###################ADDONENT applied#########################
def tf_hash_table(keys, values):
    table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(keys, values, tf.int64, tf.int64), -1)
    return table

logits = tf.constant([range(0,25), range(0, 25)], tf.float32)
basenet = tf.constant([[0.1, 0.1, 0.1, 0.1, 0.5, 0.1, 0.2], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5]])

keys = [0, 1, 2, 3, 4, 5, 6]

values_upper_padding = [6, 13, 0, 9, 18, 17, 2]
values_lower_padding = [16, 8, 23, 12, 0, 7, 19]

upper_table = tf_hash_table(keys, values_upper_padding)
lower_table = tf_hash_table(keys, values_lower_padding)
basenet_key = tf.argmax(basenet, 1) # rank1

basenet_key = tf.cast(basenet_key, tf.int64)

upper_value = upper_table.lookup(basenet_key)
upper_value = tf.cast(upper_value, tf.int32)
lower_value = lower_table.lookup(basenet_key)
lower_value = tf.cast(lower_value, tf.int32)

def lambda_concat(upper_value, lower_value, net):
    pad1 = tf.zeros([upper_value], tf.float32)
    pad2 = tf.zeros([lower_value], tf.float32)
    cropped_net = net[upper_value: 25 -lower_value]
    return tf.concat([pad1, cropped_net, pad2], 0)

preds = tf.map_fn(lambda x : lambda_concat(x[0], x[1], x[2]), (upper_value, lower_value, logits), dtype=tf.float32)

predictions = tf.argmax(preds, 1)

with tf.Session() as sess:
    upper_table.init.run(session = sess)
    lower_table.init.run(session = sess)
    print(sess.run([logits, preds, predictions]))

#labels = tf.squeeze(labels)