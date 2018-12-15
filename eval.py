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
from TextImageGeneratorBM import TextImageGeneratorBM, report_accuracy

from config import BATCH_SIZE, img_size, num_channels, label_len
import pdb

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



def main(img_dir):
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
        #ckpt_state = tf.train.get_checkpoint_state('./model/LPRtf3.ckpt-42000')

        saver.restore(session, './model/LPRtf3.ckpt-42000')
        print('model/LPRtf3.ckpt-42000 loaded!!!!')
        #test_inputs, test_targets, test_seq_len = test_gen.next_batch()
        test_feed = {inputs: images,
                     seq_len: 24}
        st = time.time()
        dd = session.run(decoded[0], test_feed)
        pdb.set_trace()




def batch_test(img_dir, label_file):

    logits, inputs, targets, seq_len = get_train_model(num_channels, label_len, BATCH_SIZE, img_size)
    logits = tf.transpose(logits, (1, 0, 2))
    # tragets是一个稀疏矩阵
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
    # tragets是一个稀疏矩阵
    loss = tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len)
    cost = tf.reduce_mean(loss)

    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=MOMENTUM).minimize(cost, global_step=global_step)
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

    # 前面说的划分块之后找每块的类属概率分布，ctc_beam_search_decoder方法,是每次找最大的K个概率分布
    # 还有一种贪心策略是只找概率最大那个，也就是K=1的情况ctc_ greedy_decoder
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        saver.restore(session, './model/LPRtf3.ckpt-42000')
        test_gen = TextImageGeneratorBM(img_dir=img_dir,
                                      label_file=label_file,
                                      batch_size=BATCH_SIZE,
                                      img_size=img_size,
                                      num_channels=num_channels,
                                      label_len=label_len)
        #do_report(test_gen, 3)
        for i in range(4):
            test_inputs, test_targets, test_seq_len = test_gen.next_batch()
            test_feed = {inputs: test_inputs,
                         targets: test_targets,
                         seq_len: test_seq_len}
            st = time.time()
            dd = session.run(decoded[0], test_feed)
            tim = time.time() - st
            print('time:%s' % tim)
            report_accuracy(dd, test_targets)



def run_lpr(img_dir, out_dir):
    input_imgs = glob.glob(img_dir + '/*.jpg')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i, image_name in enumerate(input_imgs):
        print(i, image_name)



def parse_args():
    parser = argparse.ArgumentParser(description='Plate end to end test')
    parser.add_argument('--img_dir', default='/ssd/wfei/data/testing_data/wanda_plates_v1.2',
                        type=str, help='Input test image dir')
    parser.add_argument('--out_dir', default='/ssd/wfei/data/testing_data/lpr_results_v1.0',
                        type=str, help='Output image dir')
    parser.add_argument('--label_file', default='/ssd/wfei/data/testing_data/wanda_benchmark_label.json',
                        type=str, help='Output image dir')

    args = parser.parse_args()
    return args



if __name__ == '__main__':

    args = parse_args()

    #run_lpr(args.img_dir, args.out_dir)
    #batch_test(args.img_dir, args.label_file)
    img_path = '/Users/fei/data/parking/carplate/testing_data/wanda_benchmark/wanda_plates_v1.2'
    main(args.img_dir)
