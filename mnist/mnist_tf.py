import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

img_length = 784
classes = 10
learning_rate = 1e-3
epochs = 30
batch_size = 100

x = tf.placeholder(tf.float32,[None,img_length])
y = tf.placeholder(tf.float32,[None,classes])

w = tf.Variable(tf.random_normal([img_length,classes]))
b = tf.Variable(tf.random_normal([classes]))

hypothesis = tf.nn.softmax(tf.matmul(x,w) + b)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis),axis=1))

train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

prediction = tf.argmax(hypothesis,1)
cor = tf.equal(prediction,tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(cor,tf.float32))

with tf.Session() as s:
	s.run(tf.global_variables_initializer())
	batches = int(mnist.train.num_examples / batch_size)
	for epoch in xrange(epochs):
		for batch in xrange(batches):
			train_x, train_y = mnist.train.next_batch(batch_size)
			s.run(train,feed_dict={x:train_x,y:train_y})
		print 'Epoch',epoch,'finished with cost:',s.run(cost,feed_dict={x:train_x,y:train_y})
	print '-' * 50
	acc = s.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
	print 'training finished with accuracy:',acc
