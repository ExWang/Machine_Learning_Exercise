#!/usr/bin/python

# Ref:https://confluence.slac.stanford.edu/display/PSDM/How+to+access+HDF5+data+from+Python 

import h5py
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# matlab file name
filename_mat = 'digitStruct.mat'
path_mat = 'format1/test/'
datapath_mat = path_mat + filename_mat
print datapath_mat

# TFRecords file name
filename_tfr = 'test_32x32_digit.tfrecords'
path_tfr = 'format1/'
datapath_tfr = path_tfr + filename_tfr


#
# Bounding Box
#
class BBox:
    def __init__(self):
        self.label = ""  # Digit
        self.left = 0
        self.top = 0
        self.width = 0
        self.height = 0


class DigitStruct:
    def __init__(self):
        self.name = None  # Image file name
        self.bboxList = None  # List of BBox structs


# Function for tensorflow trans
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _ano_int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_img(addr):
    img = Image.open(addr)
    img.tobytes()
    return img

# Function for debugging

def printHDFObj(theObj, theObjName):
    isFile = isinstance(theObj, h5py.File)
    isGroup = isinstance(theObj, h5py.Group)
    isDataSet = isinstance(theObj, h5py.Dataset)
    isReference = isinstance(theObj, h5py.Reference)
    print("{}".format(theObjName))
    print("    type(): {}".format(type(theObj)))
    if isFile or isGroup or isDataSet:
        # if theObj.name != None:
        #    print "    name: {}".format(theObj.name)
        print("    id: {}".format(theObj.id))
    if isFile or isGroup:
        print("    keys: {}".format(theObj.keys()))
    if not isReference:
        print("    Len: {}".format(len(theObj)))

    if not (isFile or isGroup or isDataSet or isReference):
        print(theObj)


def readDigitStructGroup(dsFile):
    dsGroup = dsFile["digitStruct"]
    return dsGroup


#
# Reads a string from the file using its reference
#
def readString(strRef, dsFile):
    strObj = dsFile[strRef]
    str = ''.join(chr(i) for i in strObj)
    return str


#
# Reads an integer value from the file
#
def readInt(intArray, dsFile):
    intRef = intArray[0]
    isReference = isinstance(intRef, h5py.Reference)
    intVal = 0
    if isReference:
        intObj = dsFile[intRef]
        intVal = int(intObj[0])
    else:  # Assuming value type
        intVal = int(intRef)
    return intVal


def yieldNextInt(intDataset, dsFile):
    for intData in intDataset:
        intVal = readInt(intData, dsFile)
        yield intVal


def yieldNextBBox(bboxDataset, dsFile):
    for bboxArray in bboxDataset:
        bboxGroupRef = bboxArray[0]
        bboxGroup = dsFile[bboxGroupRef]
        labelDataset = bboxGroup["label"]
        leftDataset = bboxGroup["left"]
        topDataset = bboxGroup["top"]
        widthDataset = bboxGroup["width"]
        heightDataset = bboxGroup["height"]

        left = yieldNextInt(leftDataset, dsFile)
        top = yieldNextInt(topDataset, dsFile)
        width = yieldNextInt(widthDataset, dsFile)
        height = yieldNextInt(heightDataset, dsFile)

        bboxList = []

        for label in yieldNextInt(labelDataset, dsFile):
            bbox = BBox()
            bbox.label = label
            bbox.left = next(left)
            bbox.top = next(top)
            bbox.width = next(width)
            bbox.height = next(height)
            bboxList.append(bbox)

        yield bboxList


def yieldNextFileName(nameDataset, dsFile):
    for nameArray in nameDataset:
        nameRef = nameArray[0]
        name = readString(nameRef, dsFile)
        yield name


# dsFile = h5py.File('../data/gsvhn/train/digitStruct.mat', 'r')
def yieldNextDigitStruct(dsFileName):
    dsFile = h5py.File(dsFileName, 'r')
    dsGroup = readDigitStructGroup(dsFile)
    nameDataset = dsGroup["name"]
    bboxDataset = dsGroup["bbox"]

    bboxListIter = yieldNextBBox(bboxDataset, dsFile)
    for name in yieldNextFileName(nameDataset, dsFile):
        bboxList = next(bboxListIter)
        obj = DigitStruct()
        obj.name = name
        obj.bboxList = bboxList
        yield obj


def testMain():
    dsFileName = 'digitStruct.mat'
    testCounter = 0
    dsFileName = datapath_mat

    writer = tf.python_io.TFRecordWriter(datapath_tfr)

    for dsObj in yieldNextDigitStruct(dsFileName):
        # testCounter += 1
        img_name = dsObj.name.split('.')[0]
        image_path = path_mat + dsObj.name
        #print img_name, ',', (dsObj.name)
        img_name = int(img_name)
        box_num = 0

        image_raw = load_img(image_path)

        bbox_list = []
        box_this = []
        for bbox in dsObj.bboxList:

            box_this.append(bbox.label)
            box_this.append(bbox.left)
            box_this.append(bbox.top)
            box_this.append(bbox.width)
            box_this.append(bbox.height)
            #bbox_list.append(box_this)
            box_num += 1

        #print box_num, '\n', box_this.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'name': _int64_feature(img_name),
            'box_num': _int64_feature(box_num),
            'bboxes': _ano_int64_feature(box_this),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
            #print("    {}-{}:{},{},{},{}".format(
            #    box_num, bbox.label, bbox.left, bbox.top, bbox.width, bbox.height))
        #if testCounter >= 5:
        #    break

        # 356MB ----> 5.9GB
        # 356MB ----> 2.7GB
    writer.close()
    print 'work finished'

if __name__ == "__main__":
    testMain()

