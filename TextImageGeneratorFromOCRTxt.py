#coding=utf-8
import os
import numpy as np
import cv2
import json
import pdb

from config import CHARS, dict, CHARS_DICT, NUM_CHARS

class TextImageGeneratorBM:
    def __init__(self, label_file, batch_size, img_size, num_channels=3, label_len=8):
        #self._img_dir = img_dir
        self._label_file = label_file
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
        self.labels = []
        plate_label = load_gt(self._label_file, 0)

        for filename, chars in plate_label.iteritems():
            if True:
                #if '\u4e00' <= label[0] <= '\u9fff':
                #    label = filename[:7]
                if len(chars) >=8:
                    self.filenames.append(filename)
                    #print(filename, chars)
                    label = encode_label(chars[:8])
                    self.labels.append(label)
                    self._num_examples += 1
                elif len(chars) == 7:
                    self.filenames.append(filename)
                    label = np.append(encode_label(chars), 73)
                    self.labels.append(label)
                    self._num_examples += 1
                else:
                    print "Skip ", filename, chars.encode('utf8'),  "!!!"
                #print label    
        print "Total number of files, ", len(self.filenames)
        print self._num_examples
        self.labels = np.float32(self.labels)
        self._num_batches = self._num_examples//self._batch_size +1



    def next_batch(self):
        # Shuffle the data
        """if self._next_index == 0:
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
        images = np.zeros([batch_size, self._img_h, self._img_w, self._num_channels])
        #images = np.zeros([end-start, self._img_h, self._img_w, self._num_channels])

        # labels = np.zeros([batch_size, self._label_len])

        for j, i in enumerate(range(start, end)):
            fname = self.filenames[i]
            img = cv2.imread(fname)
            img = cv2.resize(img, (self._img_w, self._img_h), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255.0
            img = img[:, :, np.newaxis]
            images[j, ...] = img
        images = np.transpose(images, axes=[0, 2, 1, 3])
        labels = self.labels[start:end, ...]
        targets = [np.asarray(i) for i in labels]
        sparse_labels = sparse_tuple_from(targets)
        # input_length = np.zeros([batch_size, 1])

        seq_len = np.ones(self._batch_size) * 24
        return images, sparse_labels, seq_len, self.filenames[start:end]


def load_gt(ocrtxt_file, skip):
    gts = {}
    for line in open(ocrtxt_file, 'r'):
        fname, label = line.split(';')
        #bname = os.path.basename(fname).replace('_plate.jpg', '')

        plate = label.strip().decode('utf8')
        plate = plate.replace('|', '')
        # print bname, plate
        gts[fname] = plate[skip:]
    print "Total number of gt", len(gts)
    return gts

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


def decode_sparse_tensor(sparse_tensor):
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    result = []
    for index in decoded_indexes:
        result.append(decode_a_seq(index, sparse_tensor))
    return result


def decode_a_seq(indexes, spars_tensor):
    decoded = []
    for m in indexes:
        str = CHARS[spars_tensor[1][m]]
        decoded.append(str)
    return decoded

def encode_label(s):
    label = np.zeros([len(s)])
    for i, c in enumerate(s):
        if c.encode('utf-8') in CHARS_DICT:
            label[i] = CHARS_DICT[c.encode('utf-8')]
        else:
            label[i] = 73
            print 'Label not in dict!!!', c, label
    return label



def report_accuracy(decoded_list, test_targets, scores):
    original_list = decode_sparse_tensor(test_targets)
    detected_list = decode_sparse_tensor(decoded_list)
    true_numer = 0

    if len(original_list) != len(detected_list):
        print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
              " test and detect length doesn't match")
        return
    print("T/F: original(length) <-------> detectcted(length)")
    for idx, number in enumerate(original_list):
        detect_number = detected_list[idx]
        hit = (number == detect_number)
        gt = ''.join(number).decode('utf-8')
        detect = ''.join(detect_number).decode('utf-8')
        if not hit:
            print hit, gt.encode('utf8'), "(", len(number), ") <-------> ", detect.encode('utf8'), "(", len(detect_number), ")", scores[idx]
        if hit:
            true_numer = true_numer + 1
    print("Test Accuracy:", true_numer * 1.0 / len(original_list))

    return detected_list, true_numer, len(original_list)

def write_ocr(detected_list, scores, filenames, out_dir):
    """
    Write output to the individual files so that precision and recall numbers can be evaluated
    :param detected_list:
    :param filenames:
    :param out_dir:
    :return:
    """
    assert(len(detected_list) == len(scores))
    assert(len(detected_list) == len(filenames))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for idx, number in enumerate(detected_list):
        detect_number = detected_list[idx]
        detect = ''.join(detect_number)
        score = scores[idx]
        fname = filenames[idx].split('_plate.jpg')[0]
        fname = fname.replace('.jpg', '.txt')
        fpath = os.path.join(out_dir, fname)

        out_str = ' '.join([detect, str(score)])
        with open(fpath, 'w') as f:
            f.write(out_str)
        #print fname, out_str
