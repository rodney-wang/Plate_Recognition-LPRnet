#!/usr/bin/python
# -*- coding: UTF-8 -*-
import argparse
import glob
import os
import numpy as np
import json


def file_from_folder(plate_folder):
    fnames = glob.glob(os.path.join(plate_folder, '*.jpg'))
    bnames = [os.path.basename(f) for f in fnames]
    return bnames

def main(train_list_file, gt_json, train_file):

    with open(gt_json) as f:
        chars_gt = json.load(f)
    with open(train_list_file, 'r') as f:
        train_list = f.readlines()

    print(train_list)
    fout = open(train_file, 'w')
    count = 0
    for plate_file in train_list:
        plate_file = plate_file.strip()
        if plate_file not in chars_gt:
            continue

        true_ocr = chars_gt[plate_file]
        print true_ocr
        count += 1
        out_line = plate_file + ';'
        out_label = '|' + '|'.join(true_ocr) + '|'

        out_line += out_label.encode('utf-8') + '\n'
        print(out_line)  # fout.write(out_line)
        fout.write("%s" % out_line)
    print "Training file was written to: ", train_file
    print "Total records", count

def parse_args():
    parser = argparse.ArgumentParser(description='Plate end to end test')
    parser.add_argument('--file_list',
                        default='/Users/fei/data/parking/carplate/forlabel/non_canton/train_list.txt',
                        type=str, help='list of training files')
    parser.add_argument('--gt_json',
                        default='/Users/fei/data/parking/carplate/forlabel/non_canton/non_canton_plate_ocr_gt_20190122.json',
                        type=str, help='Json file which stores the true ocr results')
    parser.add_argument('--train_file',
                        default='/Users/fei/data/parking/carplate/forlabel/non_canton/non_canton_plate_4training_20190122.txt',
                        type=str, help='Output training file in crnn training format')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    main(args.file_list, args.gt_json, args.train_file)


