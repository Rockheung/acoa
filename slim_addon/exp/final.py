import tensorflow as tf


mat = tf.Variable([ [0.7, 0.3, 0, 0], [0.2, 30, 2, 0] , [0.1, 0.7, 0.2, 0], [0.2, 30, 2, 0 ]])
index = tf.constant([1, 3, 0, 2])
c = tf.Variable([1.0, 2.0, 3.0, 4.0])
init = tf.initialize_all_variables()
with tf.Session() as sess:
	sess.run(init)
	for i in range(0, index.shape[0]):
		print i
		k = tf.assign(mat[i], c)
		sess.run(k)
	print sess.run(k)
	