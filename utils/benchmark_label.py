#coding=utf-8
import os
import argparse
import numpy as np
import cv2
import glob
import json
from cutplate import four_point_transform

def PolygonArea(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

def write_plate_label_to_json(img_path, out_dir):
    """
    Write benchmark plate label to a json file
    """
    input_folder = glob.glob(img_path + '/*.txt')
    #output_folder = os.path.join(out_dir, 'wanda')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plate_label = {}
    for i, txt_name in enumerate(input_folder):
        bname = os.path.basename(txt_name)
        print 'Processing ', i, bname

        jpg_name = bname.replace('.txt', '.jpg')

        fo = open(txt_name, "r")
        line = fo.readline()
        chars = line.split(',')[-1].strip()
        plate_label[jpg_name] = chars

    out_json = os.path.join(out_dir, 'benchmark_label.json')
    with open(out_json, 'w') as f:
        json.dump(plate_label, f)
    print 'Plate label written to : ', out_json


def write_plate_image(img_path, json_path, out_dir):

    corner = json.load(open(json_path))

    input_folder = glob.glob(img_path + '/*.jpg')
    #output_folder = os.path.join(out_dir, 'wanda')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i, image_name in enumerate(input_folder):
        print(image_name)

        if i % 10 == 0:
            print('### Processing: ', i)

        img = cv2.imread(image_name)
        #print(img.shape)
        bname = os.path.basename(image_name)
        if bname not in corner:
            continue
        res = corner[bname]
        #print(res)
        #plate_chars = res['test']
        corner_pts = res['pts']
        #print(corner_pts)
        pts = np.array(corner_pts).reshape([4, 2])
        area=PolygonArea(pts)
        img = four_point_transform(img, corner_pts)
        out_path = os.path.join(out_dir,  bname+'_plate.png')
        #if area <1600:
        #    cv2.imwrite(out_path, img)
        cv2.imwrite(out_path, img)


def parse_args():
    parser = argparse.ArgumentParser(description='Plate end to end test')
    parser.add_argument('--img_dir', default='/ssd/zq/parkinglot_pipeline/carplate/test_data/image_data',
                        type=str, help='Input test image dir')
    parser.add_argument('--corner_json', default='/ssd/zq/parkinglot_pipeline/carplate/test_data/corner_output_v1.2.json',
                        type=str, help='Corner detection results in json')
    parser.add_argument('--out_dir', default='/ssd/wfei/data/testing_data/wanda_plates_v1.2',
                        type=str, help='Output image dir')

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    json_path_corner = '/ssd/zq/parkinglot_pipeline/carplate/test_data/corner_output.json'
    json_path_corner = '/Users/fei/data/parking/carplate/testing_data/wanda_benchmark/corner_output_v1.2.json'
    img_path ='/Users/fei/data/parking/carplate/testing_data/wanda_benchmark/car_crop'
    out_dir  ='/Users/fei/data/parking/carplate/testing_data/wanda_benchmark/'
    img_path = '/Users/fei/data/parking/carplate/testing_data/k11_benchmark/car_crop'
    out_dir = '/Users/fei/data/parking/carplate/testing_data/k11_benchmark/'

    args = parse_args()
    write_plate_label_to_json(args.img_dir, args.out_dir)
    #write_plate_image(img_path, json_path_corner, out_dir)

    # To run it on K11 data
    #python benchmark_label.py --image_dir /ssd/zq/parkinglot_pipeline/carplate/test_data/image_data
    # --corner_json /ssd/zq/parkinglot_pipeline/carplate/test_data_k11/corner_output_v1.2.json
    # --out_dir /ssd/wfei/data/testing_data/k11_plates_v1.2
    #write_plate_label_to_json(img_path, out_dir)
    #write_plate_image(args.img_dir, args.corner_json, args.out_dir)
