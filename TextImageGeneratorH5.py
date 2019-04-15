#coding=utf-8
import os
import numpy as np
import cv2
import os
import glob
import h5py
from augment_data import augment_data

from config import CHARS, CHARS_DICT

class generator:
    def __init__(self, file):
        self.file = file

    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            for im in hf["data"]:
                yield im

def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


class TextImageGeneratorH5:
    def __init__(self, h5_path, batch_size, img_size, num_channels=1, label_len=8):
        self._h5_files = glob.glob(os.path.join(h5_path, '*.h5'))
        self._batch_size = batch_size
        self._num_channels = num_channels
        self._label_len = label_len
        self._img_w, self._img_h = img_size

        self._num_examples = 0
        self._next_index = 0
        self._num_epoches = 0
        self._num_batches = 0

        self.filenames = []
        self.labels = []

        self.init()

    def init(self):
        ns =[]

        for i, filename in enumerate(self._h5_files):
            ff = h5py.File(filename, 'r')
            ns.append( ff["data"].shape[0] )
            self.labels.extend(ff["labels"])
        self._num_examples = sum(ns)
        print "Number of sample images:", ns
        print "Total images:", self._num_examples
        sh = h5py.File(self._h5_files[0], 'r')["data"].shape  # get the first ones shape.

        self.X = h5py.VirtualLayout(shape=(self._num_examples,) + sh, dtype=np.uint8)
        for i, filename in enumerate(self._h5_files):
            vsource = h5py.VirtualSource(filename, 'sh', shape=sh)
            self.X[i, :, :, :] = vsource

        self._num_batches = self._num_examples//self._batch_size +1

    def next_batch(self):
        # Shuffle the data
        """if self._next_index %78 == 0:
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._filenames = [self.filenames[i] for i in perm]
            self._labels = self.labels[perm]
        """

        batch_size = self._batch_size
        start = self._next_index
        end = self._next_index + batch_size
        if end > self._num_examples:
            self._next_index = 0
            #start = self._next_index
            #end = self._next_index + batch_size
            start = self._num_examples - batch_size
            end = self._num_examples
            self._num_epoches += 1
        else:
            self._next_index = end

        #labels = np.zeros([batch_size, self._label_len])
        images = np.zeros([batch_size, self._img_h, self._img_w, self._num_channels])
        for i,j in enumerate(range(start, end)):
            img = np.squeeze(self.X[i, ...])
            img = augment_data(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images[j, ...] = img[..., np.newaxis]

        images = np.transpose(images, axes=[0, 2, 1, 3])
        labels = self._labels[start:end, ...]
        targets = [np.asarray(i) for i in labels]
        sparse_labels = sparse_tuple_from(targets)
        # input_length = np.zeros([batch_size, 1])

        seq_len = np.ones(self._batch_size) * 24
        return images, sparse_labels, seq_len


if __name__ == '__main__':
    h5_path = '/ssd/wfei/code/Plate_Recognition-LPRnet/data/lpr_train_color'
    img_size = [94, 24]

    train_gen = TextImageGeneratorH5(img_dir=h5_path,
                                   batch_size=128,
                                   img_size=img_size,
                                   num_channels=1,
                                   label_len=24)

