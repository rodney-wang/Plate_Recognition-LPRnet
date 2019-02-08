#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import os
import sys
import glob
import json
import argparse
from shutil import copyfile

sys.path.append('../lprnew')
from config_new import CHARS, dict

dict_chi = dict
print(len(dict_chi))

chi_str_dict = {v.decode('utf-8'):k for k, v in dict_chi.items()}
print(chi_str_dict)

def convert_chars(plate_chars):
    res=[]
    plen = len(plate_chars)
    for i, char in enumerate(plate_chars):
        if char == 'I':
            char = '1'
        if char == 'O':
            char = '0'
        if char is not u'':
            if char in chi_str_dict:
                if i == 0:
                    res.append(chi_str_dict[char])
                    res.append('_')
                else:
                    res.append('_')
                    res.append(chi_str_dict[char])
            else:
                res.append(char)
    return ''.join(res)

def copy_file(src_path, tgt_path):

    if os.path.exists(tgt_path):
        return
    cmd = "cp " + src_path + ' ' + tgt_path
    print cmd
    try:
        copyfile(src_path, tgt_path) 
    except:
        print('### Special plate detected ###')
        pass


def batch_rename_copy(filename, tgt_folder):
    data = pd.read_csv(filename, sep=';', encoding='utf8', error_bad_lines=False, header=None,
                       names=['path', 'transcription'], escapechar='\\')

    if not os.path.exists(tgt_folder):
        os.makedirs(tgt_folder)

    # store the number of counts of each plate
    plate_count ={}
    for index, row in data.iterrows():
        #if index > 10:
        #   continue
        plate_path = row.path
        plate_chars = row.transcription.split('|')
        plate_chars = [c for c in plate_chars if c is not u'']
        print plate_chars
        pid = convert_chars(plate_chars)
        print plate_chars, pid
        if pid not in plate_count:
            plate_count[pid] = 0
        else:
            plate_count[pid] += 1

        tgt_name = pid + '_' + str(plate_count[pid]).zfill(3) + '.jpg'
        print index, tgt_name

        copy_file(plate_path, os.path.join(tgt_folder, tgt_name))


def batch_benchmark_rename_copy(json_file, src_folder, tgt_folder):
    plate_label = json.load(open(json_file))

    if not os.path.exists(tgt_folder):
        os.makedirs(tgt_folder)

    input_images= glob.glob(src_folder + '/*.png')
    plate_count = {}
    for index, image_name in enumerate(input_images):
        print(image_name)

        plate_path = image_name
        bname = os.path.basename(plate_path).split('_plate.png')[0]
    
        #plate_chars = plate_label
        label = plate_label[bname]
        pid = convert_chars(list(label))
        #print plate_chars, pid
        if pid not in plate_count:
            plate_count[pid] = 0
        else:
            plate_count[pid] += 1

        tgt_name = pid + '_' + str(plate_count[pid]).zfill(3) + '.png'
        print index, tgt_name

        copy_file(plate_path, os.path.join(tgt_folder, tgt_name))


def parse_args():
    parser = argparse.ArgumentParser(description='Plate end to end test')
    parser.add_argument('--train_csv',
                        default='/Users/fei/data/parking/carplate/forlabel/non_canton/non_canton_plate_4training_20190122.txt',
                        type=str, help='list of training files')
    parser.add_argument('--tgt_folder',
                        default='/Users/fei/data/parking/carplate/forlabel/non_canton/non_canton_train_lpr',
                        type=str, help='Json file which stores the true ocr results')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    batch_rename_copy(args.train_csv, args.tgt_folder)



