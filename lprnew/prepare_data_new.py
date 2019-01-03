#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import os
import glob
import json
from shutil import copyfile
from config_new import CHARS, dict

# train_csv = "/Users/fei/tmp/wanda_plates_1105/20181105_plates_crnn_training_100k.txt"
# train_csv = "/ssd/zq/parkinglot_pipeline/carplate/data/k11_plates_for_training_1219.txt"
# train_csv = "/ssd/wfei/data/ocr_training/20181008_plate_and_label_filtered.txt"
# train_csv = "/ssd/wfei/data/ocr_training/20181121_plate_and_label_filtered_crnn6.txt"
# train_csv = "/ssd/zq/parkinglot_pipeline/carplate/data/20181220_crnn_training_data_label_v1.8e"
train_csv = "/ssd/zq/parkinglot_pipeline/carplate/data/20181206_crnn_training_data_label_v1.7"

# dict_chi = {'A01':'京','A02':'津','A03':'沪','B02':'蒙',
#         'S01':'皖','S02':'闽','S03':'粤','S04':'甘',
#         'S05': '贵', 'S06': '鄂', 'S07': '冀', 'S08': '黑', 'S09': '湘',
#         'S10': '豫', 'S12': '吉', 'S13': '苏', 'S14': '赣', 'S15': '辽',
#         'S17': '川', 'S18': '鲁', 'S22': '浙',
#         'S30':'渝', 'S31':'晋', 'S32':'桂', 'S33':'琼', 'S34':'云', 'S35':'藏',
#         'S36':'陕','S37':'青', 'S38':'宁', 'S39':'新'}

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


tgt_folder = '/ssd/wfei/data/LPR_training/20181206_crnn_data_train_v1.7_new'
#tgt_folder = '/Users/fei/tmp/test/tgt_folder'

batch_rename_copy(train_csv, tgt_folder)

# json_file = '/ssd/wfei/data/testing_data/wanda_benchmark_label.json'
# src_dir  ='/ssd/wfei/data/testing_data/wanda_plates_v1.2'
# tgt_dir = '/ssd/wfei/data/testing_data/wanda_plates_v1.2_with_label'
# batch_benchmark_rename_copy(json_file, src_dir, tgt_dir)


