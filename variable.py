import tensorflow as tf
import numpy as np

state = tf.Variable(0,name="test")
#print(state.name)

two = tf.constant(1)

new_value = tf.add(state,two)
update = tf.assign(state,new_value)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for _ in range(3):
	sess.run(update)
	print(sess.run(state))