import tensorflow as tf


mat = tf.Variable([ [0.7, 0.3, 0, 0], [0.2, 30, 2, 0] , [0.1, 0.7, 0.2, 0], [0.2, 30, 2, 0 ]])
index = tf.constant([1, 3, 0, 2])

with tf.Session() as sess:
	for i in range(0, index.shape[0]):
		tf.assign(mat[i][i], index[i])

	