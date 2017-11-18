import tensorflow as tf

slim = tf.contrib.slim

cons = tf.constant(0)
encoded = slim.one_hot_encoding(cons, 10)
#new = tf.zeros
# padding = tf.zeros([num], tf.int32)
# result = tf.concat([padding, ])
x = tf.argmax(encoded)


with tf.Session() as sess:
	print(sess.run(x))

	# ```python
 #    t1 = [[1, 2, 3], [4, 5, 6]]
 #    t2 = [[7, 8, 9], [10, 11, 12]]
 #    tf.concat([t1, t2], 0) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
 #    tf.concat([t1, t2], 1) ==> [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
    
 #    # tensor t3 with shape [2, 3]
 #    # tensor t4 with shape [2, 3]
 #    tf.shape(tf.concat([t3, t4], 0)) ==> [4, 3]
