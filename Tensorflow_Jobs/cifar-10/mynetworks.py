# -*- coding: utf-8 -*-
import tensorflow as tf
import os
from PIL import Image

# My works
import mycifar_inputer as mycf


dataset_bin_dir_main = mycf.dataset_bin_dir_main
batchsize = 1000


image, label = mycf.input_distorted(dataset_bin_dir_main, batchsize, flag_test=False)  #training set
height = mycf.fixed_height
width = mycf.fixed_width
train_samples_per_epoch = mycf.train_set_per_epoch
test_samples_per_epoch = mycf.test_set_per_epoch


def weight_variable(name, shape, normal, WeightDecay=None):
    initial = tf.truncated_normal_initializer(shape, stddev=normal)
    if WeightDecay is not None:
        initial = tf.multiply(tf.nn.l2_loss(initial), WeightDecay, name='weight_loss')  # lambda * sigma (theta^2)
        tf.add_to_collection('losses', initial)
    return tf.Variable(name=name, variable_def=initial)


def bias_variable(name, shape):
    initial = tf.zeros(shape)  # tf.constant(0.1, shape=shape)
    return tf.Variable(name=name, variable_def=initial)



def netbuild(image):

    ''' 每次输入一个batch的 64 幅图像， 转化成 64*32*32*3 的四维张量，经过步长为 1，卷积核大小为 5*5 ，
    Feature maps 为64的卷积操作，变为 64*32*32*64 的四维张量，然后经过一个步长为 2 的 max_pool 的池化层，
    变成 64*16*16*64 大小的四维张量，再经过一次类似的卷积池化操作，
    变为 64*8*8*64 大小的4维张量，再经过两个全连接层，映射到 64*192 的二维张量，然后经过一个 sortmax 层，
    变为 64*10 的张量，最后和标签 label 做一个交叉熵的损失函数'''
    with tf.variable_scope('conv1') as scope:
        kernel = weight_variable('weights', [5, 5, 3, 64], normal=0.1)  # 5x5 conv 64
        biases = bias_variable('biases', [64])
        wa = tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1], padding='SAME')  # BECAUSE
        z = wa + biases
        conv1 = tf.nn.relu(z, name=scope.name)
        print conv1.shape
        tf.summary.histogram(scope.name+'/activations', conv1)

    with tf.variable_scope('pool+norm') as scope:
        max_pool = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool_1')
        # ksize[1,height,width,1]    |   strides[1,height,width,1]      maybe 2x2 is better
        







with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    image_show = image
    for i in range(10):
        image_show = tf.cast(image_show, tf.uint8)
        example, l = sess.run([image_show, label])
        print example.shape, len(example)
        print l.shape, len(l)
        # img = Image.fromarray(example, 'RGB')
        # img.save(str(i) + '_''Label_' + str(l) + '.jpg')
        # print(example, l)
    coord.request_stop()
    coord.join(threads)









