import tensorflow as tf

net = tf.Variable([ [0.7, 0.3, 0, 0, 0.3, 0, 0], [0.2, 30, 2, 0, 0.2, 30, 2] , [0.1, 0.7, 0.2, 0, 0.1, 0.7, 0.2], [0.2, 30, 2, 0.2, 30, 2, 0 ]])


temp = tf.zeros([5], tf.float32)

sess = tf.Session()
init = tf.initialize_all_variables()

sess.run(init)

a = tf.concat([net[0], temp], 0)

print sess.run(a)

print sess.run([tf.rank(net[0]), tf.rank(temp)])