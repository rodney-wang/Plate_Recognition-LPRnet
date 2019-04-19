#!/usr/bin/env python
# coding=utf-8
import os
import numpy as np
from multiprocessing import Process
import h5py
import cv2
import json
import random
import argparse
import codecs

#CAFFE_ROOT = os.getcwd()   # assume you are in $CAFFE_ROOT$ dir
IMAGE_WIDTH, IMAGE_HEIGHT = 94, 24
LABEL_SEQ_LEN = 8
char_dict = json.load(open('utils/carplate.json', 'r'))
num_dict =  {v: k for k, v in char_dict.iteritems()}

def write_image_info_into_file(file_name, data_tuple):
    with codecs.open(file_name, 'w', encoding='utf-8') as f:
        for datum in data_tuple:
            img_path, numbers = datum
            numbers_str = [str(num) for num in numbers]
            chars = [num_dict.get(i, '-') for i in numbers]
            f.write(img_path + "|" + ','.join(chars) + "\n")


def write_image_info_into_hdf5(file_name, data_tuple, phase):
    total_size = len(data_tuple)
    print '[+] total image for {0} is {1}'.format(file_name, len(data_tuple))
    single_size = 20000
    groups = total_size / single_size
    if total_size % single_size:
        groups += 1
    def process(file_name, data):
        img_data = np.zeros((len(data_tuple), IMAGE_HEIGHT, IMAGE_WIDTH), dtype = np.float32)
        label_seq = 73*np.ones((len(data_tuple), LABEL_SEQ_LEN), dtype = np.float32)
        for i, datum in enumerate(data_tuple):
            img_path, numbers = datum
            label_seq[i, :len(numbers)] = numbers
            img = cv2.imread(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) #load as grayscale
            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT)) 
            #img = caffe.io.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH, 1))
            #img = np.transpose(img, (2, 0, 1))
            img_data[i] = img
            #"""
            if (i+1) % 1000 == 0:
                print '[+] ###{} name: {}'.format(i, img_path)
                print '[+] number: {}'.format(','.join(map(lambda x: str(x), numbers)))
                print '[+] label: {}'.format(','.join(map(lambda x: str(x), label_seq[i])))
            #"""
        with h5py.File(file_name, 'w') as f:
            f.create_dataset('data', data = img_data)
            f.create_dataset('label', data = label_seq)
    with open(file_name, 'w') as f:
        workspace = os.path.split(file_name)[0]
        process_pool = []
        for g in xrange(groups):
            h5_file_name = os.path.join(workspace, '%s_%d.h5' %(phase, g))
            f.write(h5_file_name + '\n')
            start_idx = g*single_size
            end_idx = start_idx + single_size
            if g == groups - 1:
                end_idx = len(data_tuple)
            p = Process(target = process, args = (h5_file_name, data_tuple[start_idx:end_idx]))
            p.start()
            process_pool.append(p)
        for p in process_pool:
            p.join()

def write_h5(train_csv, h5_path):

    images, labels =[], []
    count =0
    for line in open(train_csv, 'r'):
        #if count >100:
        #    break 
        line.strip()
        img_path, label = line.split(';')
        label = label.strip()[1:-1]
        numbers = [char_dict.get(c.decode('utf-8'), 73) for c in label.split('|')]
        if len(numbers)>8:
            print "Warning: plate length more than 8, skip!!!"
            print img_path, label, numbers
            continue 
        images.append(img_path)
        labels.append(numbers)
        count += 1
    print '[+] total image number: {}'.format(len(images))

    data_all = list(zip(images, labels))
    random.shuffle(data_all)

    #trainning_size = 86000   # number of images for trainning
    trainning_data = data_all
    #testing_data = data_all[:2000]
    #write_image_info_into_hdf5(os.path.join(h5_path, 'plate_trainning.list'), trainning_data, 'train')
    #write_image_info_into_hdf5(os.path.join(h5_path, 'plate_testing.list'), testing_data, 'test')


def parse_args():
    parser = argparse.ArgumentParser(description='Convert the labeling csv files into h5 files for caffe training')
    parser.add_argument('--train_csv', default='/mnt/soulfs2/wfei/code/crnn.caffe/data/plate/train_ocrlabel/k11_train_20190408.txt',
                        type=str, help='Image path and labels in CRNN txt labeling file format')
    parser.add_argument('--h5_path', default='/ssd/wfei/code/Plate_Recognition-LPRnet/data/',
                        type=str, help='Path to write the h5 file and list file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    write_h5(args.train_csv, args.h5_path)
