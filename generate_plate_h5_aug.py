#!/usr/bin/env python
# coding=utf-8
import os
import numpy as np
from multiprocessing import Process
#import caffe
import cv2
import h5py
import json
import random
import argparse
import codecs
from augment_data import augment_data
import pdb

#CAFFE_ROOT = os.getcwd()   # assume you are in $CAFFE_ROOT$ dir
IMAGE_WIDTH, IMAGE_HEIGHT = 94, 24
LABEL_SEQ_LEN = 8
char_dict = json.load(open('/mnt/soulfs2/wfei/code/crnn.caffe/examples/lprnet/utils/carplate.json', 'r'))
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
        img_data = np.zeros((len(data_tuple), 1, IMAGE_HEIGHT, IMAGE_WIDTH), dtype = np.float32)
        label_seq = 73*np.ones((len(data_tuple), LABEL_SEQ_LEN), dtype = np.float32)
        for i, datum in enumerate(data_tuple):
            img_path, numbers, do_aug = datum
            label_seq[i, :len(numbers)] = numbers

            img = cv2.imread(img_path)
            if img is None: 
                continue

            if do_aug:
                img = augment_data(img)
                if img is None: 
                    continue
            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
            #img = img[..., np.newaxis]
            #img = img/255.
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
            print '=== H5 data written to ', file_name
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

def write_h5(train_csv, h5_path, prefix, list_name, do_aug):

    images, labels, aug =[], [], []
    count =0
    for line in open(train_csv, 'r'):
        #if count >100:
        #    break
        if count % 10000 ==0: 
            print count, line
        line.strip()
        img_path, label = line.split(';')
        label = label.strip()[1:-1]
        numbers = [char_dict.get(c.decode('utf-8'), 73) for c in label.split('|')]
        if len(numbers)>8:
            print "Plate label too long!!!",img_path, label, numbers 
            continue

        #if not os.path.exists(img_path):
        #    print "None exist", img_path
        #    return 
        images.append(img_path)
        labels.append(numbers)
        aug.append(False)
        count += 1

        if do_aug:
            images.append(img_path)
            labels.append(numbers)
            aug.append(True)
            count += 1
    print '[+] total image number: {}'.format(len(images))

    data_all = list(zip(images, labels, aug))
    random.shuffle(data_all)

    #trainning_size = 182000   # number of images for trainning
    #trainning_data = data_all[:trainning_size]
    trainning_data = data_all

    testing_data = data_all[:2048]
    write_image_info_into_hdf5(os.path.join(h5_path, list_name), trainning_data, prefix)
    #write_image_info_into_hdf5(os.path.join(h5_path, "plate_testing_wanda.list"), testing_data, 'wanda_test')


def parse_args():
    parser = argparse.ArgumentParser(description='Convert the labeling csv files into h5 files for caffe training')
    parser.add_argument('--train_csv', default='/ssd/zq/parkinglot_pipeline/carplate/data/20181206_crnn_training_data_label_v1.7_k11A500',
                        type=str, help='Image path and labels in CRNN txt labeling file format')
    parser.add_argument('--h5_path', default='/mnt/soulfs2/wfei/code/crnn.caffe/data/plate/crnn',
                        type=str, help='Path to write the h5 file and list file')
    parser.add_argument('--prefix', default='train_aug', type=str, help='h5 file prefix')
    parser.add_argument('--list_name', default='plate_trainning_aug.list', type=str, help='list filename containing the list of h5 files')
    parser.add_argument('--aug', action='store_true', help='whether to perform data augmentation')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    write_h5(args.train_csv, args.h5_path, args.prefix, args.list_name, args.aug)
