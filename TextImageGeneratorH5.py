#coding=utf-8
import os
import numpy as np
import cv2
import os
import glob
import h5py
import tensorflow as tf

from config import CHARS, CHARS_DICT

class generator:
    def __init__(self, file):
        self.file = file

    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            for im in hf["data"]:
                yield im

with tf.Session() as sess:
    batch_size = 128
    hdf5_path = '/ssd/wfei/code/Plate_Recognition-LPRnet/data/lpr_train'
    filenames = glob.glob(os.path.join(hdf5_path, '*.h5'))

    ds = tf.data.Dataset.from_tensor_slices(filenames)

    ds = ds.interleave(lambda filename: tf.data.Dataset.from_generator(
            generator(filename),
            tf.uint8,
            tf.TensorShape([24,94])),
            1, batch_size)

    value = ds.make_one_shot_iterator().get_next()

# Example on how to read elements
while True:
    try:
        data = sess.run(value)
        print(data.shape)
    except tf.errors.OutOfRangeError:
        print('done.')
        break
