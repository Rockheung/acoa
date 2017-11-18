

def hash_table(key, addon):
	
	  keys = tf.constant(lv2_id_to_lv1_id.keys(), dtypes.int64)
      values = tf.constant(lv2_id_to_lv1_id.values(), dtypes.int64)

      # a hash table is defined for translating
      table = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(keys, values, dtypes.int64, dtypes.int64), -1
      )
      out = table.lookup(label)

      #conditional operator in tensorflow
      label = tf.cond(tf.equal(condition, tf.constant(1)), lambda : out, lambda : label)