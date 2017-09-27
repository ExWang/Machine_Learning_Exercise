# -*- coding: utf-8 -*-
import tensorflow as tf
import os
from PIL import Image

# My works
import mycifar_inputer as mycf


dataset_bin_dir_main = mycf.dataset_bin_dir_main
train_log_dir = dataset_bin_dir_main + 'log/'


batchsize = 64
NUM_CLASSES = 10
learning_rate = 0.1
iterations = 10000


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


def net_build(image):

    """ 每次输入一个batch的 64 幅图像， 转化成 64*32*32*3 的四维张量，经过步长为 1，卷积核大小为 5*5 ，
    Feature maps 为64的卷积操作，变为 64*32*32*64 的四维张量，然后经过一个步长为 2 的 max_pool 的池化层，
    变成 64*16*16*64 大小的四维张量，再经过一次类似的卷积池化操作，
    变为 64*8*8*64 大小的4维张量，再经过两个全连接层，映射到 64*192 的二维张量，然后经过一个 sortmax 层，
    变为 64*10 的张量，最后和标签 label 做一个交叉熵的损失函数"""
    with tf.variable_scope('conv1') as scope:
        kernel = weight_variable('weights', [5, 5, 3, 64], normal=0.005)  # 5x5 conv 64
        biases = bias_variable('biases', [64])
        wa = tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1], padding='SAME')
        z = wa + biases
        conv1 = tf.nn.relu(z, name=scope.name)
        print conv1
        tf.summary.histogram(scope.name+'/activations', conv1)

    with tf.variable_scope('pool+norm_1') as scope:
        max_pool_1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool_1')
        # ksize[1,height,width,1]    |   strides[1,height,width,1]      maybe 2x2 is better
        norm_1 = tf.nn.lrn(max_pool_1, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm_1')
        # local response normalization all parameter from the formula

    with tf.variable_scope('conv2') as scope:
        kernel = weight_variable('weights', [5, 5, 64, 64], normal=0.005)
        biases = bias_variable('biases', [64])
        wa = tf.nn.conv2d(norm_1, kernel, strides=[1, 1, 1, 1], padding='SAME')
        z = wa + biases
        conv2 = tf.nn.relu(z, name=scope.name)
        print conv2
        tf.summary.histogram(scope.name + '/activations', conv2)

    with tf.variable_scope('pool+norm_2') as scope:
        norm_2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm_1')
        # local response normalization all parameter from the formula
        max_pool_2 = tf.nn.max_pool(norm_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool_1')
        # ksize[1,height,width,1]    |   strides[1,height,width,1]      maybe 2x2 is better

    # 再经过两个全连接层，映射到 64*192 的二维张量,

    with tf.variable_scope('full_connected_1') as scope:
        reshape = tf.reshape(max_pool_2, [batchsize, -1])
        dim = reshape.get_shape()[1].value  # '-1' means caculate automatic
        weights = weight_variable('weights', shape=[dim, 384], normal=0.04, WeightDecay=0.004)
        biases = bias_variable('biases', [384])
        wa = tf.matmul(reshape, weights)
        z = wa + biases
        full_connected_1 = tf.nn.relu(z, name=scope.name)
        tf.summary.histogram(scope.name + '/activations', full_connected_1)

    with tf.variable_scope('full_connected_2') as scope:
        weights = weight_variable('weights', shape=[384, 192], normal=0.04, WeightDecay=0.004)
        biases = bias_variable('biases', [384])
        wa = tf.matmul(full_connected_1, weights)
        z = wa + biases
        full_connected_2 = tf.nn.relu(z, name=scope.name)
        tf.summary.histogram(scope.name + '/activations', full_connected_2)

    # 然后经过一个 softmax 层，变为 64*10 的张量，
    with tf.variable_scope('softmax_linear') as scope:
        weights = weight_variable('weights', shape=[192, NUM_CLASSES], normal=1/192.0, WeightDecay=0)
        biases = bias_variable('biases', shape=[NUM_CLASSES])
        softmax_linear = tf.matmul(full_connected_2, weights) + biases
        tf.summary.histogram(scope.name + '/activations', softmax_linear)

    return softmax_linear

# 最后和标签 label 做一个交叉熵的损失函数
def losses(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
    # just got the vector, if compute loss need to do reduce_mean  |  [batch_size, 1]
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def train_net():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, name='global_step', trainable=False)
        image, label = mycf.input_distorted(dataset_bin_dir_main, batchsize, flag_test=False)  # training set

        net = net_build(image)
        loss = losses(image, label)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        train_op = optimizer.minimize(loss=loss, global_step=global_step)

        saver = tf.train.Saver(tf.all_variables())

        summary_op = tf.summary.merge_all()



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








