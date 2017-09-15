import os
import sys

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# matlab file name
filename_mat = 'extra_32x32.mat'
path_mat = 'format2/'
datapath_mat = path_mat + filename_mat
print datapath_mat

# TFRecords file name
filename_tfr = 'extra_32x32.tfrecords'
path_tfr = 'format2/'
datapath_tfr = path_tfr + filename_tfr

data_mat = sio.loadmat(datapath_mat)

plt.close('all')
X_mat = data_mat['X']
y_mat = data_mat['y']

'''
print 'X(i):\n', type(X_mat), len(X_mat)
print X_mat
print 'y(i):\n', type(y_mat), len(y_mat)
print y_mat
'''

# print X_mat[:, :, :, 1], y_mat[1]

images = X_mat
labels = y_mat
labels_num = len(y_mat)

img_height = images.shape[0]
img_width = images.shape[1]
img_depth = images.shape[2]
img_nums = images.shape[3]
num_examples = 0
print img_height, img_width, img_depth, img_nums

images = images.transpose((3, 0, 1, 2))

if img_nums != labels_num:
    print '\n\033[1;31mERROR: Images num is not equal to lables num.\033[0m'
    exit(1)
else:
    num_examples = labels_num

writer = tf.python_io.TFRecordWriter(datapath_tfr)

for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(img_height),
        'width': _int64_feature(img_width),
        'depth': _int64_feature(img_depth),
        'label': _int64_feature(int(labels[index])),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
writer.close()

print '============================'
print 'Target(mat):', datapath_mat
print 'Total(TFRecords):', datapath_tfr
print 'Work finished!'
print '============================'
