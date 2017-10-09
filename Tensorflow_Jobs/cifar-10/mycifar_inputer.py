import tensorflow as tf
import os
import numpy as np
from PIL import Image

import cPickle

IMAGE_SIZE_HEIGHT = 32
IMAGE_SIZE_WIDTH = 32

expanded_height = 224
expanded_width = 224

fixed_height = 24
fixed_width = 24

train_set_per_epoch = 50000
test_set_per_epoch = 10000

bach_size = 128

defalut_step1_reshapbox = [3, 32, 32]
defalut_step2_transbox = [1, 2, 0]

dataset_bin_dir_main = 'cifar-10-batches-bin/'
dataset_py_dir_main = 'cifar-10-batches-py/'


def load_data(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
        print file, '--->Loaded successfully!'
    return dict


def read_cifar10_bin(filedir, isTest):
    dataset_name_main = 'data_batch_'
    index = 1
    dataset_path_list = []
    if not isTest:
        for index in range(1, 6):
            dataset_name = dataset_name_main + str(index) + '.bin'
            dataset_path = filedir + dataset_name
            dataset_path_list.append(dataset_path)
        print dataset_path_list
    else:
        dataset_name = 'test_batch.bin'
        dataset_path = filedir + dataset_name
        dataset_path_list.append(dataset_path)
        print dataset_path_list
    filename_queue = tf.train.string_input_producer(string_tensor=dataset_path_list)
    lab = load_data(dataset_py_dir_main + 'batches.meta')
    # dataset = load_data(dataset_path_list[0])
    # data = dataset['data']
    # labels = dataset['labels']

    num_cases_per_batch = lab['num_cases_per_batch']
    label_names = lab['label_names']
    num_vis = lab['num_vis']
    print lab

    Bytes2Read = num_vis + 1  # image + label
    print 'Bytes_to_read-->', Bytes2Read
    reader = tf.FixedLengthRecordReader(record_bytes=Bytes2Read)
    key, value_str = reader.read(filename_queue)

    value_uint8 = tf.decode_raw(bytes=value_str, out_type=tf.uint8)

    label = tf.strided_slice(value_uint8, [0], [1])
    label = tf.cast(label, tf.int32)

    image_raw = tf.strided_slice(value_uint8, [1], [1 + num_vis])
    image_raw = tf.reshape(image_raw, defalut_step1_reshapbox)
    image_raw = tf.transpose(image_raw, defalut_step2_transbox)

    print 'image_raw-->', image_raw, '\nlabel-->', label

    return image_raw, label


def read_cifar10_py(filedir):
    dataset_name_main = 'data_batch_'
    index = 1
    dataset_path_list = []
    for index in range(1, 6):
        dataset_name = dataset_name_main + str(index)
        dataset_path = filedir + dataset_name
        dataset_path_list.append(dataset_path)
    print dataset_path_list
    filename_queue = tf.train.string_input_producer(string_tensor=dataset_path_list)
    lab = load_data(filedir + 'batches.meta.txt')
    # dataset = load_data(dataset_path_list[0])
    # data = dataset['data']
    # labels = dataset['labels']
    num_cases_per_batch = lab['num_cases_per_batch']
    label_names = lab['label_names']
    num_vis = lab['num_vis']
    print lab

    Bytes2Read = num_vis + 1  # image + label
    print 'Bytes_to_read-->', Bytes2Read
    reader = tf.FixedLengthRecordReader(record_bytes=Bytes2Read)
    key, value_str = reader.read(filename_queue)

    value_uint8 = tf.decode_raw(bytes=value_str, out_type=tf.uint8)

    label = tf.strided_slice(value_uint8, [0], [1])
    label = tf.cast(label, tf.int32)

    image_raw = tf.strided_slice(value_uint8, [1], [1 + num_vis])
    image_raw = tf.reshape(image_raw, defalut_step1_reshapbox)
    image_raw = tf.transpose(image_raw, defalut_step2_transbox)
    image_raw = tf.cast(image_raw, tf.uint8)

    print 'image_raw-->', image_raw, 'label-->', label

    return image_raw, label


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.summary.image('input_images', images)

    return images, tf.reshape(label_batch, [batch_size])


def input_distorted(data_dir, batch_size, flag_test):  # Image processing for training the network. Note the many random
    image_raw, label = read_cifar10_bin(data_dir, isTest=flag_test)
    reshaped_image = tf.cast(image_raw, tf.float32)
    tf.summary.image('raw_input_image', tf.reshape(reshaped_image, [1, 32, 32, 3]))

    height = fixed_height
    width = fixed_width

    if not flag_test:
        distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

        float_image = tf.image.per_image_standardization(distorted_image)

    else:
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, height, width)
        float_image = tf.image.per_image_standardization(resized_image)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])
    label.set_shape([1])

    if flag_test:
        examples_per_epoch = test_set_per_epoch
    else:
        examples_per_epoch = train_set_per_epoch

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(examples_per_epoch *
                             min_fraction_of_examples_in_queue)
    print ('Filling queue with %d CIFAR images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)
    batch_image, batch_label = _generate_image_and_label_batch(float_image, label,
                                                               min_queue_examples, batch_size,
                                                               shuffle=True)
    return batch_image, batch_label


def input_24x(data_dir, batch_size,flag_test):    # Image processing for evaluation.
    image_raw, label = read_cifar10_bin(data_dir, isTest=flag_test)
    reshaped_image = tf.cast(image_raw, tf.float32)
    tf.summary.image('raw_input_image', tf.reshape(reshaped_image, [1, 32, 32, 3]))

    height = fixed_height
    width = fixed_width


    # Crop the central [height, width] of the image.
    distorted_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           height, width)

    float_image = tf.image.per_image_standardization(distorted_image)
    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])
    label.set_shape([1])

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(train_set_per_epoch *
                             min_fraction_of_examples_in_queue)
    print ('Filling queue with %d CIFAR images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)
    batch_image, batch_label = _generate_image_and_label_batch(float_image, label,
                                                               min_queue_examples, batch_size,
                                                               shuffle=True)
    return batch_image, batch_label

def input_32x_org(data_dir, batch_size,flag_test):    # Image processing for evaluation.
    image_raw, label = read_cifar10_bin(data_dir, isTest=flag_test)
    reshaped_image = tf.cast(image_raw, tf.float32)
    tf.summary.image('raw_input_image', tf.reshape(reshaped_image, [1, 32, 32, 3]))

    height = fixed_height
    width = fixed_width


    # Crop the central [height, width] of the image.
    distorted_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           height, width)

    float_image = tf.image.per_image_standardization(distorted_image)
    # Set the shapes of tensors.
    float_image.set_shape([32, 32, 3])
    label.set_shape([1])

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(train_set_per_epoch *
                             min_fraction_of_examples_in_queue)
    print ('Filling queue with %d CIFAR images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)
    batch_image, batch_label = _generate_image_and_label_batch(float_image, label,
                                                               min_queue_examples, batch_size,
                                                               shuffle=True)
    return batch_image, batch_label

if __name__ == "__main__":

    # image_raw, label = read_cifar10_bin(dataset_bin_dir_main, isTest=False)
    # image_float32 = tf.cast(image_raw, tf.float32)
    batchsize = 1000
    image, label = input_distorted(dataset_bin_dir_main, batchsize)

    # height = fixed_height
    # width = fixed_width

    # image_resized = tf.image.resize_image_with_crop_or_pad(image_float32, height, width)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        image_show = image
        for i in range(10):
            image_show = tf.cast(image_show, tf.uint8)
            example, l = sess.run([image_show, label])
            print example.shape,len(example)
            print l.shape,len(l)
            #img = Image.fromarray(example, 'RGB')
            #img.save(str(i) + '_''Label_' + str(l) + '.jpg')
            # print(example, l)
        coord.request_stop()
        coord.join(threads)

