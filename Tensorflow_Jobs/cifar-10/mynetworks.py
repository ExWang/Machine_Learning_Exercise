# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import math
from PIL import Image

# My works
import mycifar_inputer as mycf


dataset_bin_dir_main = mycf.dataset_bin_dir_main
train_log_dir = dataset_bin_dir_main + 'log/'


BATCH_SIZE = 64
NUM_CLASSES = 10
LEARNING_RATE = 0.1
ITERATIONS = 10000
ITER_SHOW = 30
ITER_SUMMARY = 100
ITER_CKPT_SAVE = 1000

FLAG_RUN_ONCE = False


height = mycf.fixed_height
width = mycf.fixed_width
train_samples_per_epoch = mycf.train_set_per_epoch
test_samples_per_epoch = mycf.test_set_per_epoch


def weight_variable(name, shape, normal, WeightDecay=None):
    dtype = tf.float32
    initializer = tf.truncated_normal_initializer(stddev=normal, dtype=dtype)
    var = tf.get_variable(name='weight', shape=shape, initializer=initializer, dtype=dtype)
    if WeightDecay is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), WeightDecay, name='weight_loss')  # lambda * sigma (theta^2)
        tf.add_to_collection('losses', weight_decay)
    return var


def bias_variable(name, shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(name=name, initial_value=initial)


def net_build(image):

    """ 每次输入一个batch的 64 幅图像， 转化成 64*32*32*3 的四维张量，经过步长为 1，卷积核大小为 5*5 ，
    Feature maps 为64的卷积操作，变为 64*32*32*64 的四维张量，然后经过一个步长为 2 的 max_pool 的池化层，
    变成 64*16*16*64 大小的四维张量，再经过一次类似的卷积池化操作，
    变为 64*8*8*64 大小的4维张量，再经过两个全连接层，映射到 64*192 的二维张量，然后经过一个 sortmax 层，
    变为 64*10 的张量，最后和标签 label 做一个交叉熵的损失函数"""
    with tf.variable_scope('conv1') as scope:
        kernel = weight_variable('weights', [5, 5, 3, 64], normal=0.005)  # 5x5 conv 64
        print kernel
        biases = bias_variable('biases', [64])
        wa = tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1], padding='SAME')
        z = wa + biases
        conv1 = tf.nn.relu(z, name=scope.name)
        print 'conv1:', conv1
        tf.summary.histogram(scope.name+'/activations', conv1)

    with tf.variable_scope('pool_norm_1') as scope:
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

    with tf.variable_scope('pool_norm_2') as scope:
        norm_2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm_1')
        # local response normalization all parameter from the formula
        max_pool_2 = tf.nn.max_pool(norm_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool_1')
        # ksize[1,height,width,1]    |   strides[1,height,width,1]      maybe 2x2 is better

    # 再经过两个全连接层，映射到 64*192 的二维张量,

    with tf.variable_scope('full_connected_1') as scope:
        reshape = tf.reshape(max_pool_2, [BATCH_SIZE, -1])
        dim = reshape.get_shape()[1].value  # '-1' means caculate automatic
        weights = weight_variable('weights', shape=[dim, 384], normal=0.04, WeightDecay=0.004)
        biases = bias_variable('biases', [384])
        wa = tf.matmul(reshape, weights)
        z = wa + biases
        full_connected_1 = tf.nn.relu(z, name=scope.name)
        tf.summary.histogram(scope.name + '/activations', full_connected_1)

    with tf.variable_scope('full_connected_2') as scope:
        weights = weight_variable('weights', shape=[384, 192], normal=0.04, WeightDecay=0.004)
        biases = bias_variable('biases', [192])
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

    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
    # just got the vector, if compute loss need to do reduce_mean  |  [batch_size, 1]
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def train_net():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, name='global_step', trainable=False)
        image, label = mycf.input_distorted(dataset_bin_dir_main, BATCH_SIZE, flag_test=False)  # training set

        net = net_build(image)
        loss = losses(net, label)

        tf.summary.scalar('loss', loss)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)

        train_op = optimizer.minimize(loss=loss, global_step=global_step)

        saver = tf.train.Saver(tf.all_variables())

        summary_op = tf.summary.merge_all()  # generate all summary data

        init_op = tf.initialize_all_variables()

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.9

        with tf.Session(config=config) as sess:
            sess.run(init_op)

            ckpt = tf.train.get_checkpoint_state(train_log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Restore from ckpt file success!')
            else:
                print('No ckpt file found!')

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)

            try:
                for step in xrange(ITERATIONS):
                    start_time = time.time()
                    _, loss_value = sess.run([train_op, loss])
                    duration = time.time() - start_time

                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                    if step % ITER_SHOW == 0:
                        num_examples_per_step = BATCH_SIZE
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = float(duration)
                        print
                        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                      'sec/batch)')
                        print (format_str % (datetime.datetime.now(), step, loss_value,
                                             examples_per_sec, sec_per_batch))

                    if step % ITER_SUMMARY == 0:
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, global_step=step)

                    if step % ITER_CKPT_SAVE == 0 or (step+1) == ITERATIONS:
                        checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                        saver.save(sess=sess, save_path=checkpoint_path, global_step=step)

            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)

            sess.close()
            print('Finished!')


def evaluate():
    with tf.Graph().as_default() as g:
        image, labels = mycf.input_distorted(dataset_bin_dir_main, BATCH_SIZE, flag_test=True)  # training set

        logits = net_build(image)

        # Calculate predictions.If all predictions in targets,out like tis [True, False, False ........]
        top_k_op = tf.nn.in_top_k(predictions=logits, targets=labels, k=1)

        saver = tf.train.Saver(tf.all_variables())

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.9

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(train_log_dir, g)

        while True:

            with tf.Session() as sess:

                ckpt = tf.train.get_checkpoint_state(train_log_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    print('Restore from ckpt file success!'+'\nglobal_step is:'+global_step)
                else:
                    print('No ckpt file found!')
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                try:
                    true_count = 0
                    step = 0
                    iterations = int(math.ceil(mycf.test_set_per_epoch / BATCH_SIZE))
                    print ('>', iterations, '<')
                    while step < iterations and not coord.should_stop():
                        predictions = sess.run([top_k_op])
                        true_count += np.sum(predictions)
                        step += 1

                    accuracy = float(true_count) / float(mycf.test_set_per_epoch)
                    print('++++++++++++++++++++++++++++++++++++++')
                    print('%s: Accuracy @ 1 = %.3f' % (datetime.datetime.now(), accuracy))
                    print('++++++++++++++++++++++++++++++++++++++')

                    summary = tf.Summary()
                    summary.ParseFromString(sess.run(summary_op))
                    summary.value.add(tag='Precision @ 1', simple_value=accuracy)
                    summary_writer.add_summary(summary, global_step)

                except tf.errors.OutOfRangeError as e:
                    coord.request_stop(e)
                finally:
                    coord.request_stop()
                    coord.join(threads)
            sess.close()

            if FLAG_RUN_ONCE:
                break

            time.sleep(2)


if __name__ == "__main__":
    # train_net()
    evaluate()
    print('Finished!')

'''

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

'''
