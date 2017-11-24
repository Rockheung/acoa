import tensorflow as tf

in_ = tf.constant([[1, 2, 3, 4], [2, 2, 3, 4], [5, 2, 3, 4]])
padding = [[0, 0], [1, 1]]
padded = tf.pad(in_[:,0:2], padding, "CONSTANT")

with tf.Session() as sess:
	print(sess.run(padded))
