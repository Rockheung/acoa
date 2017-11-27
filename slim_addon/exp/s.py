import tensorflow as tf

# _input = tf.Variable([[1, 2], [3, 4]])
# index = tf.constant([1, 2])

# def tmp(x, y):
# 	return x + y

# final_result = tf.map_fn(tmp, _input, index)

# init = tf.initialize_all_variables()
a = tf.constant([[1,2,3],[4,5,6]])
b = tf.constant([1, 2])
b = tf.reshape(b, [2, 1])
sess = tf.Session()
#print sess.run(b)

c = a + b
print sess.run(c)
#c = tf.map_fn(lambda x, y: x+y, a,b)

# #sess.run(init)

# print sess.run(c)