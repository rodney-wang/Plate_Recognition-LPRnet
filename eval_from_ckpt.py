#coding=utf-8
import os
import argparse
import numpy as np
import sys
import cv2
import glob
import tensorflow as tf
import time
from model import get_train_model
from TextImageGeneratorBM import TextImageGeneratorBM, report_accuracy, write_ocr

#from config_new import BATCH_SIZE, img_size, num_channels, label_len, NUM_CHARS
from config import  img_size, num_channels, label_len, NUM_CHARS
BATCH_SIZE = 256

import pdb
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def get_images(input_path):
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(input_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files

def resize_image(img):
    '''
    resize image to a fixed size []
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    _img_w, _img_h = img_size
    img = cv2.resize(img, (_img_w, _img_h), interpolation=cv2.INTER_CUBIC)
    return img



def main(img_dir, lpr_model):
    files = get_images(img_dir)
    img = cv2.imread(files[1])
    img = resize_image(img)

    images = img[np.newaxis, :]
    images = np.transpose(images, axes=[0, 2, 1, 3])
    
    global_step = tf.Variable(0, trainable=False)
    logits, inputs, targets, seq_len = get_train_model(num_channels, label_len, 1, img_size)
    logits = tf.transpose(logits, (1, 0, 2))

    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)

    saver = tf.train.Saver()
    config = tf.ConfigProto(device_count = {'GPU': 1})
    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        saver.restore(session, lpr_model)
        print("{} loaded!!!".format(lpr_model))
        #test_inputs, test_targets, test_seq_len = test_gen.next_batch()
        test_feed = {inputs: images,
                     seq_len: 24}
        #st = time.time()
        #dd = session.run(decoded[0], test_feed)
        lg = session.run(logits, test_feed)

        #pdb.set_trace()



def batch_eval(img_dir, label_file, out_dir):

    global_step = tf.Variable(0, trainable=False)
    logits, inputs, targets, seq_len = get_train_model(num_channels, label_len, BATCH_SIZE, img_size)
    logits = tf.transpose(logits, (1, 0, 2))
    # tragets是一个稀疏矩阵
    #decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
    # tragets是一个稀疏矩阵
    loss = tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len)
    cost = tf.reduce_mean(loss)

    #decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits,seq_len,
                                                      merge_repeated=False,
                                                      beam_width=100,
                                                      top_paths=3)
    score = tf.subtract(log_prob[:, 0], log_prob[:, 1], name='score_computation')

    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        #saver.restore(session, './model/LPRk110k.ckpt-72000')
        #saver.restore(session, './model/LPRMore.ckpt-48000')
        #saver.restore(session, './model_h5/LPR_energy_c1.ckpt-72000')
        saver.restore(session, './model_wanda/LPR_wanda.ckpt-80000')

        test_gen = TextImageGeneratorBM(img_dir=img_dir,
                                      label_file=label_file,
                                      batch_size=BATCH_SIZE,
                                      img_size=img_size,
                                      num_channels=num_channels,
                                      label_len=label_len)
        nbatches = test_gen._num_batches
        print('### Number of batches = {}'.format(nbatches))
        for i in range(nbatches):
            test_inputs, test_targets, test_seq_len, img_names = test_gen.next_batch()
            test_feed = {inputs: test_inputs,
                         #targets: test_targets,
                         seq_len: test_seq_len}
            st = time.time()
            [dd, probs, scores] = session.run([decoded[0], log_prob, score], test_feed)
            tim = time.time() - st
            print('time:%s' % tim)
            #print(scores)
            detected_list = report_accuracy(dd, test_targets, scores)
            write_ocr(detected_list, scores, img_names, out_dir)



def parse_args():
    parser = argparse.ArgumentParser(description='Plate end to end test')
    parser.add_argument('--img_dir', default='/ssd/wfei/data/testing_data/k11_plates_v1.2',
                        type=str, help='Input test image dir')
    parser.add_argument('--out_dir', default='/ssd/wfei/data/testing_data/k11_tfresults_lpr_v2.0',
                        type=str, help='Output image dir')
    parser.add_argument('--label_file', default='/ssd/wfei/data/testing_data/k11_benchmark_label.json',
                        type=str, help='Output image dir')

    args = parser.parse_args()
    return args



if __name__ == '__main__':

    args = parse_args()

    batch_eval(args.img_dir, args.label_file, args.out_dir)
