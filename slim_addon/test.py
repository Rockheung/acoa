import tensorflow as tf
import numpy as np
from tensorflow.python.framework import dtypes

# data = range(11)
# np_data = np.array(data)

# tensors = tf.constant(np_data)
# node1 = tf.add(tensors, tf.constant(1, dtypes.int64))

# branch1 = tf.multiply(node1, tf.constant(10, dtype=dtypes.int64))
# branch2 = node1
# branch3 = tf.add(node1, tf.constant(10, dtypes.int64))
# branch1 = tf.multiply(branch1, tf.constant(10, dtype=dtypes.int64))
# branch1 = tf.multiply(branch1, tf.constant(10, dtype=dtypes.int64))
branch1 = tf.constant(1)
branch2 = tf.constant(2)
#tf.reshape()

scope_name = 'test'
with tf.variable_scope("test"):
	w1 = tf.get_variable(name="weight1", dtype=tf.int32, initializer=tf.constant(2))
	net = tf.multiply(branch1, w1)
	new = net
	w2 = tf.get_variable(name="weight2", dtype=tf.int32, initializer=tf.constant(3))
	net = tf.multiply(branch1, w2)


with tf.variable_scope("check"):
	new = tf.multiply(branch2, new, name='multi')

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)
print sess.run(new)

print new
