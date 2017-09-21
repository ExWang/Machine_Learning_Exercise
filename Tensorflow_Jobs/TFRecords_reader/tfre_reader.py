# -*- coding: utf-8 -*-

import tensorflow as tf
from PIL import Image

default_filename = 'format2/test_32x32_te0.tfrecords'
default_dir = ''
default_box = [32, 32, 3]


def read_format2(filename, isFull=None, isShow=None):  # 读入tfrecords
    """
    :param filename: File to read.
    :param isFull: If fully print information.
    :param isShow: If show some pictures to judge the validity of reader.
    :return: image, label
    """
    filename_queue = tf.train.string_input_producer([filename])  # 生成一个queue队列

    feature = {
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string)}
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example, feature)  # 将image数据和label取出来

    img = tf.decode_raw(features['image_raw'], tf.uint8)  # 将字符串解析成图像对应的像素组  tf.float32
    reshape_box = default_box
    img = tf.reshape(img, reshape_box)  # reshape图片

    image = tf.cast(img, tf.uint8)  # * (1. / 255) - 0.5  # 在流中抛出img张量
    label = tf.cast(features['label'], tf.int32)  # 在流中抛出label张量

    if isFull:
        print 'Image is:', image
        print 'Label is:', label

    if isShow:
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            for i in range(10):
                example, l = sess.run([image, label])
                img = Image.fromarray(example, 'RGB')
                img.save(default_dir + str(i) + '_''Label_' + str(l) + '.jpg')
                # print(example, l)
            coord.request_stop()
            coord.join(threads)

    print 'Read work finished successful!'
    return image, label


if __name__ == "__main__":

    tf_filename = default_filename
    img, label = read_format2(tf_filename, True,True)

