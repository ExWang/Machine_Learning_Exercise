import os
import numpy
import tensorflow as tf
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

# Parameters

resize_height = 224
resize_width = 224
image_expand_way = Image.NORMAL  # Image.ANTIALIAS

# DataSets

MNIST_dir = 'MNIST_data/'
train_images = 'train-images.idx3-ubyte'
train_labels = 'train-labels.idx1-ubyte'
test_images = 't10k-images.idx3-ubyte'
test_labels = 't10k-labels.idx1-ubyte'


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_mnist(isTest):
    mnist = input_data.read_data_sets(MNIST_dir, dtype=tf.uint8, one_hot=True)
    if isTest:
        images = mnist.test.images
        labels = mnist.test.labels
        num_examples = mnist.test.num_examples
    else:
        images = mnist.train.images
        labels = mnist.train.labels
        num_examples = mnist.train.num_examples

    pixels = images.shape[1]

    print len(images), images.shape
    print len(labels), labels.shape
    print pixels
    print num_examples

    return images, labels, pixels, num_examples


def trans2tfrecord(isTest, isShowSimple):
    TFRname = ''
    if isTest == False:
        TFRname = 'train_' + str(resize_height) + 'x' + str(resize_width) + '.tfrecords'
    if isTest == True:
        TFRname = 'test_' + str(resize_height) + 'x' + str(resize_width) + '.tfrecords'
    print TFRname
    writer = tf.python_io.TFRecordWriter(TFRname)

    images, labels, pixels, num_examples = load_mnist(isTest)
    for index in range(num_examples):

        image_org = images[index]
        # print type(image_org), image_org.shape

        image_org = image_org[:, np.newaxis]
        # print type(image_org), image_org.shape

        image_org = image_org.reshape(28, 28)
        # print type(image_org), image_org.shape

        image_image = Image.fromarray(image_org)
        image_expand = image_image.resize((resize_height, resize_height), image_expand_way)

        if isShowSimple:
            plt.imshow(image_expand, interpolation='nearest')
            image_expand.show()
            break

        image_raw = image_expand.tobytes()  # 28*28=784 ----(expand)----> 224*224=50176
        image_pixels = len(image_raw)
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_raw),
            'label': _int64_feature(np.argmax(labels[index]))
        }))
        writer.write(example.SerializeToString())

    writer.close()


def trans2original(isTest, isShowSimple, num_epochs):
    reader = tf.TFRecordReader()
    TFRname = ''
    if isTest == False:
        TFRname = 'train_' + str(resize_height) + 'x' + str(resize_width) + '.tfrecords'
    if isTest == True:
        TFRname = 'test_' + str(resize_height) + 'x' + str(resize_width) + '.tfrecords'
    print TFRname
    filename_queue = tf.train.string_input_producer([TFRname], num_epochs=num_epochs)

    # one simple
    _, serialized_example = reader.read(filename_queue)

    # analysis it
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    images = tf.decode_raw(features['image_raw'], tf.uint8)
    images = tf.reshape(images, [224, 224])
    images = tf.cast(images, tf.float32) * (1. / 255) - 0.5

    labels = tf.cast(features['label'], tf.int32)

    if isShowSimple:
        sess = tf.Session()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(3):
            image, label = sess.run([images, labels])
            image_decode = image[:, np.newaxis]
            image_decode = image_decode.reshape(resize_height, resize_width)
            image_image = Image.fromarray(image_decode)
            image_image.show()

            print image.shape
            print label

    return images, labels


def input_mnist_batch(data_set, batch_size, num_epochs):
    if not num_epochs:
        num_epochs = None
    if data_set == 'train':
        isTest = False
    else:
        isTest = True

    # load from my func
    image, label = trans2original(isTest=isTest, isShowSimple=False, num_epochs=num_epochs)

    images, labels = tf.train.shuffle_batch([image, label],
                                            batch_size=batch_size,
                                            num_threads=64,
                                            capacity=1000+3*batch_size,
                                            min_after_dequeue=1000
                                            )
    isShowSimple = False

    if isShowSimple:
        sess = tf.Session()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(1):
            image, label = sess.run([images, labels])
            print len(image), image.shape, type(image)
            image_image = Image.fromarray(image)
            image_image.show()

            print image.shape
            print label

    return images, labels


if __name__ == '__main__':
    isTest = True
    # trans2tfrecord(isTest=isTest, isShowSimple=False)
    # print 'TFRecords transformed complete!'
    # trans2original(isTest=isTest, isShowSimple=True)
    batch = input_mnist_batch(data_set='train', batch_size=32, num_epochs=None)

    init = tf.initialize_all_variables()  #
    sess = tf.Session()
    threads = tf.train.start_queue_runners(sess=sess)  ####
    images = sess.run(batch[0])
    print images
    print images[0]
    print 'Works done!'
