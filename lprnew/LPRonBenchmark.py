import cv2
import os
import glob
import time
import tensorflow as tf
import argparse
import numpy as np
from tensorflow.contrib import predictor

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

"""
Test SavedModel by loading the model and then 
process all plates in benchmark
"""

def run_crnn_and_write_result(plate_file, out_dir, predict_fn, num_channel):

    plate_file = plate_file.strip()

    if not os.path.exists(plate_file.encode('utf-8')):
        print('File does not exist')
        return
    img = cv2.imread(plate_file)
    #print img.shape
    #img = cv2.resize(img, (94, 24), interpolation=cv2.INTER_CUBIC)
    img = cv2.resize(img, (94, 24))
    if num_channel == 1:
       img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       images = img[np.newaxis, :, :, np.newaxis]
    else:
       images = img[np.newaxis, :]
    
    images = np.transpose(images, axes=[0, 2, 1, 3])

    inputs = {"Placeholder": images,
              "Placeholder_4": [24]}
              #"is_train": False}

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
        print(fname, out_str.encode('utf-8'))
        #with open(out_file, 'w', encoding='utf-8') as ff:
        with open(out_file, 'w') as ff:
            ff.write(out_str.encode('utf-8'))
    return True

def batch_lpr_benchmark(img_dir, out_dir, num_channel):

    fnames = glob.glob(os.path.join(img_dir, '*.png'))
    fnames = sorted(fnames)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    MODEL_DIR = './model_pb_c1'
    predict_fn = predictor.from_saved_model(MODEL_DIR)
    start_time = time.time()
    for plate_file in fnames:
        run_crnn_and_write_result(plate_file, out_dir, predict_fn, num_channel)
    print("--- %s seconds ---" % (time.time() - start_time))

def parse_args():
    parser = argparse.ArgumentParser(description='Plate Segmentation')
    parser.add_argument('--img_dir', default='/ssd/wfei/data/testing_data/k11_plates_v1.2',
                        type=str, help='Input test image dir')
    parser.add_argument('--out_dir', default='/ssd/wfei/data/testing_data/k11_lpr_results69_saved_k11',
                        type=str, help='Output image dir')
    parser.add_argument('--num_channel', default=3,
                                type=int, help='Number of channels for the input to the ocr model')
    args = parser.parse_args()
    return args



if __name__ == '__main__':

    args = parse_args()
    batch_lpr_benchmark(args.img_dir, args.out_dir, args.num_channel)
