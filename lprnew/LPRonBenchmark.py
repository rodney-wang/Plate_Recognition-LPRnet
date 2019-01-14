import cv2
import os
import glob
import tensorflow as tf
import argparse
import numpy as np
from tensorflow.contrib import predictor

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

"""
Test SavedModel by loading the model and then 
process all plates in benchmark
"""

def run_crnn_and_write_result(plate_file, out_dir, predict_fn):

    plate_file = plate_file.strip()

    if not os.path.exists(plate_file.encode('utf-8')):
        print('File does not exist')
        return
    img = cv2.imread(plate_file)
    print img.shape
    img = cv2.resize(img, (94, 24), interpolation=cv2.INTER_CUBIC)
    images = img[np.newaxis, :]
    images = np.transpose(images, axes=[0, 2, 1, 3])

    inputs = {"Placeholder": images,
              "Placeholder_4": [24]}

    predictions = predict_fn(inputs)
    print "Prediction:      ", predictions['code2str_conversion/predicted'][0].decode('utf-8')
    print "Confidence Score:", predictions['confidence_score'][0]
    chars = predictions['code2str_conversion/predicted'][0].decode('utf-8')
    score = predictions['confidence_score'][0]

    if score != -1 and len(chars) != 0:
        fname = os.path.basename(plate_file).split('_plate.png')[0]
        fname = fname.replace('.jpg', '.txt')

        out_file = os.path.join(out_dir, fname)

        out_str = ' '.join([chars, str(score)])
        print(out_str.encode('utf-8'))
        #with open(out_file, 'w', encoding='utf-8') as ff:
        with open(out_file, 'w') as ff:
            ff.write(out_str.encode('utf-8'))
    return True

def batch_lpr_benchmark(img_dir, out_dir):

    fnames = glob.glob(os.path.join(img_dir, '*.png'))
    fnames = sorted(fnames)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    MODEL_DIR = './model_pb'
    predict_fn = predictor.from_saved_model(MODEL_DIR)

    for plate_file in fnames:
        run_crnn_and_write_result(plate_file, out_dir, predict_fn)


def parse_args():
    parser = argparse.ArgumentParser(description='Plate Segmentation')
    parser.add_argument('--img_dir', default='/ssd/wfei/data/testing_data/k11_plates_v1.2',
                        type=str, help='Input test image dir')
    parser.add_argument('--out_dir', default='/ssd/wfei/data/testing_data/k11_lpr_results69_saved_k11',
                        type=str, help='Output image dir')

    args = parser.parse_args()
    return args



if __name__ == '__main__':

    args = parse_args()
    batch_lpr_benchmark(args.img_dir, args.out_dir)
