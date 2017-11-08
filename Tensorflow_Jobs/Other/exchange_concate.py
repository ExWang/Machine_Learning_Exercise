import input_mnist
from nets import resnet_v1
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image

import tensorflow as tf 
import numpy as np 

batch_size = 32
x = tf.placeholder(tf.float32, shape = [batch_size, 224, 224, 1])
y_ = tf.placeholder(tf.float32, shape = [batch_size, 10])
net, endpoints = resnet_v1.resnet_v1_50(x, 10)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=net))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


with tf.Session(config=config) as sess:
  init = tf.initialize_all_variables()
  sess.run(init)
  batch = input_mnist.input_mnist_batch(data_set='train', batch_size=batch_size, num_epochs=None, isShuffle=False)
  threads = tf.train.start_queue_runners(sess=sess)  ####
  for i in range(20000):
    images, labels = sess.run([batch[0], batch[1]])

    images = images[:, np.newaxis]
    images = np.transpose(images, (0, 2, 3, 1))
    labels = labels[:, np.newaxis]
    marks  = np.zeros(shape = [batch_size, 10])
    for j in range(batch_size):
        marks[j][labels[j]] = 1
    labels = marks
    # print images.shape
    # print labels.shape
    if i % 100 == 0:
      train_accuracy = sess.run(accuracy, feed_dict={
          x: images, y_: labels})

      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: images, y_: labels})
    print i

#   print('test accuracy %g' % accuracy.eval(feed_dict={
#       x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

