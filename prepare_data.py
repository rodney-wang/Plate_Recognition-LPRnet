#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import os
import glob
import json

train_csv = "/Users/fei/tmp/wanda_plates_1105/20181105_plates_crnn_training_100k.txt"
train_csv = "/ssd/zq/parkinglot_pipeline/carplate/data/20181206_crnn_training_data_label_v1.7"

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z','_'
         ]
print(len(CHARS))
dict_chi = {'A01':'京','A02':'津','A03':'沪','B02':'蒙',
        'S01':'皖','S02':'闽','S03':'粤','S04':'甘',
        'S05': '贵', 'S06': '鄂', 'S07': '冀', 'S08': '黑', 'S09': '湘',
        'S10': '豫', 'S12': '吉', 'S13': '苏', 'S14': '赣', 'S15': '辽',
        'S17': '川', 'S18': '鲁', 'S22': '浙',
        'S30':'渝', 'S31':'晋', 'S32':'桂', 'S33':'琼', 'S34':'云', 'S35':'藏',
        'S36':'陕','S37':'青', 'S38':'宁', 'S39':'新'}
print(len(dict_chi))

chi_str_dict = {v.decode('utf-8'):k for k, v in dict_chi.items()}
print(chi_str_dict)

def convert_chars(plate_chars):
    res=[]
    for char in plate_chars:
        if char is not u'':
            if char in chi_str_dict:
                res.append(chi_str_dict[char])
                res.append('_')
            else:
                res.append(char)
    return ''.join(res)

def copy_file(src_path, tgt_path):

    if os.path.exists(tgt_path):
        return
    cmd = "cp " + src_path + ' ' + tgt_path
    try:
        os.system(cmd.decode('utf-8'))
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
        if index < 6085:
           continue 
        plate_path = row.path
        plate_chars = row.transcription.split('|')
        pid = convert_chars(plate_chars)
        #print plate_chars, pid
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
        bname = os.path.basename(plate_path).split('_plate.png')

        #plate_chars = plate_label
        pid = plate_label[bname]
        #print plate_chars, pid
        if pid not in plate_count:
            plate_count[pid] = 0
        else:
            plate_count[pid] += 1

        tgt_name = pid + '_' + str(plate_count[pid]).zfill(3) + '.png'
        print index, tgt_name

        copy_file(plate_path, os.path.join(tgt_folder, tgt_name))


tgt_folder = '/ssd/wfei/data/CRNN_training/20181206_crnn_data_train_v1.7'
#batch_rename_copy(train_csv, tgt_folder)

json_file = '/ssd/wfei/data/testing_data/wanda_benchmark_label.json'
src_dir  ='/ssd/wfei/data/testing_data/wanda_plates_v1.2'
tgt_dir = '/ssd/wfei/data/testing_data/wanda_plates_v1.2_with_label'

batch_benchmark_rename_copy(json_file, src_dir, tgt_dir)


