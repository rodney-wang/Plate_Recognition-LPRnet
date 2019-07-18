#coding=utf-8
import tensorflow as tf
import numpy as np
import time
import cv2
import os
import re
import random
from model import get_train_model
from TextImageGeneratorH5 import TextImageGeneratorH5, sparse_tuple_from

from config import CHARS, dict, CHARS_DICT, NUM_CHARS

os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

#训练最大轮次
num_epochs = 100

#初始化学习速率
INITIAL_LEARNING_RATE = 1e-3
DECAY_STEPS = 4000
LEARNING_RATE_DECAY_FACTOR = 0.9  # The learning rate decay factor
MOMENTUM = 0.9

#输出字符串结果的步长间隔
REPORT_STEPS = 4000

#训练集的数量
BATCH_SIZE = 256
TRAIN_SIZE = 161000 
BATCHES = TRAIN_SIZE//BATCH_SIZE
test_num = 3

ti = '/ssd/wfei/code/Plate_Recognition-LPRnet/data/wanda_train_0604'         #训练集位置
vi = '/ssd/wfei/code/Plate_Recognition-LPRnet/data/lpr_test'         #验证集位置
img_size = [94, 24]
tl = None
vl = None
num_channels = 1  
label_len = 8 


def encode_label(s):
    label = np.zeros([len(s)])
    for i, c in enumerate(s):
        label[i] = CHARS_DICT[c]
    return label

def decode_sparse_tensor(sparse_tensor):
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    result = []
    for index in decoded_indexes:
        result.append(decode_a_seq(index, sparse_tensor))
    return result


def decode_a_seq(indexes, spars_tensor):
    decoded = []
    for m in indexes:
        str = CHARS[spars_tensor[1][m]]
        decoded.append(str)
    return decoded

def small_basic_block(x,im,om):
    x = conv(x,im,int(om/4),ksize=[1,1])
    x = conv(x,int(om/4),int(om/4),ksize=[3,1],pad='SAME')
    x = conv(x,int(om/4),int(om/4),ksize=[1,3],pad='SAME')
    x = conv(x,int(om/4),om,ksize=[1,1])
    return x

def conv(x,im,om,ksize,stride=[1,1,1,1],pad = 'SAME'):
    conv_weights = tf.Variable(
        tf.truncated_normal([ksize[0], ksize[1], im, om],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=None, dtype=tf.float32))
    conv_biases = tf.Variable(tf.zeros([om], dtype=tf.float32))
    out = tf.nn.conv2d(x,
                        conv_weights,
                        strides=stride,
                        padding=pad)
    relu = tf.nn.bias_add(out, conv_biases)
    return relu


def train(a):

    train_gen = TextImageGeneratorH5(h5_path=ti,
                                   batch_size=BATCH_SIZE,
                                   img_size=img_size,
                                   num_channels=num_channels,
                                   label_len=label_len)

    val_gen = TextImageGeneratorH5(h5_path=vi,
                                 batch_size=BATCH_SIZE,
                                 img_size=img_size,
                                 num_channels=num_channels,
                                 label_len=label_len)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                               global_step,
                                               DECAY_STEPS,
                                               LEARNING_RATE_DECAY_FACTOR,
                                               staircase=True)

    print "LEARNING RATE!!!"
    logits, inputs, targets, seq_len = get_train_model(num_channels, label_len, BATCH_SIZE, img_size, True, True)
    logits = tf.transpose(logits, (1, 0, 2))
    # tragets是一个稀疏矩阵
    loss = tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len)
    cost = tf.reduce_mean(loss)

    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=MOMENTUM).minimize(cost, global_step=global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    print "After Optimizer!!!"

    # 前面说的划分块之后找每块的类属概率分布，ctc_beam_search_decoder方法,是每次找最大的K个概率分布
    # 还有一种贪心策略是只找概率最大那个，也就是K=1的情况ctc_ greedy_decoder
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False, top_paths=3)

    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))
    print "After ACC!!!"

    init = tf.global_variables_initializer()
    print "After Global TensorFlow Init!!!"

    def report_accuracy(decoded_list, test_targets):
        original_list = decode_sparse_tensor(test_targets)
        detected_list = decode_sparse_tensor(decoded_list)
        true_numer = 0

        if len(original_list) != len(detected_list):
            print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
                  " test and detect length desn't match")
            return
        print("T/F: original(length) <-------> detectcted(length)")
        for idx, number in enumerate(original_list):
            detect_number = detected_list[idx]
            hit = (number == detect_number)
            #print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
            if hit:
                true_numer = true_numer + 1
        print("Test Accuracy:", true_numer * 1.0 / len(original_list))

    def do_report(val_gen,num):
        for i in range(num):
            test_inputs, test_targets, test_seq_len = val_gen.next_batch()
            test_feed = {inputs: test_inputs,
                        targets: test_targets,
                        seq_len: test_seq_len}
            st =time.time()
            dd= session.run(decoded[0], test_feed)
            tim = time.time() -st
            print('time:%s'%tim)
            report_accuracy(dd, test_targets)

    def test_report(testi,files):
        true_numer = 0
        num = files//BATCH_SIZE

        for i in range(num):
            test_inputs, test_targets, test_seq_len = val_gen.next_batch()
            test_feed = {inputs: test_inputs,
                        targets: test_targets,
                        seq_len: test_seq_len}
            dd = session.run([decoded[0]], test_feed)
            original_list = decode_sparse_tensor(test_targets)
            detected_list = decode_sparse_tensor(dd)
            for idx, number in enumerate(original_list):
                detect_number = detected_list[idx]
                hit = (number == detect_number)
                if hit:
                    true_numer = true_numer + 1
        print("Test Accuracy:", true_numer * 1.0 / files)


    def do_batch(train_gen,val_gen):
        train_inputs, train_targets, train_seq_len = train_gen.next_batch()

        feed = {inputs: train_inputs, targets: train_targets, seq_len: train_seq_len}

        b_loss, b_targets, b_logits, b_seq_len, b_cost, steps, _ = session.run(
            [loss, targets, logits, seq_len, cost, global_step, optimizer], feed)

        #print(b_cost, steps)
        if steps > 0 and steps % REPORT_STEPS == 0:
            do_report(val_gen,test_num)
            saver.save(session, "./model_k11/LPR_wanda.ckpt", global_step=steps)
        return b_cost, steps

    with tf.Session() as session:
        session.run(init)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        if a=='train':
             start_epoch = 0
             #checkpoint = './model_wanda/LPR_wanda.ckpt-80000'
             #saver.restore(session, checkpoint)
             checkpoint_id = 0
             #start_epoch = checkpoint_id // BATCHES
             for curr_epoch in range(start_epoch, start_epoch+num_epochs):
                print("Epoch.......", curr_epoch)
                train_cost = train_ler = 0
                for batch in range(BATCHES):
                    start = time.time()
                    c, steps = do_batch(train_gen,val_gen)
                    train_cost += c * BATCH_SIZE
                    seconds = time.time() - start
                    #print("Step:", steps, ", batch seconds:", seconds)

                train_cost /= TRAIN_SIZE
                val_cs=0
                val_ls =0
                for i in range(test_num):
                    train_inputs, train_targets, train_seq_len = val_gen.next_batch()
                    val_feed = {inputs: train_inputs,
                                targets: train_targets,
                                seq_len: train_seq_len}

                    val_cost, val_ler, lr, steps = session.run([cost, acc, learning_rate, global_step], feed_dict=val_feed)
                    val_cs+=val_cost
                    val_ls+=val_ler

                log = "Epoch {}/{}, steps = {}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}s, learning_rate = {}"
                print(log.format(curr_epoch + 1, num_epochs, steps, train_cost, train_ler, val_cs/test_num, val_ls/test_num,
                                 time.time() - start, lr))
        if a =='test':
            testi='/ssd/wfei/code/Plate_Recognition-LPRnet/data/lpr_test_color'
            saver.restore(session, './model_h5/LPR_energy_c1.ckpt-30000')
            test_gen = TextImageGeneratorH5(h5_path=testi,
                                           batch_size=BATCH_SIZE,
                                           img_size=img_size,
                                           num_channels=num_channels,
                                           label_len=label_len)
            do_report(test_gen,4)


if __name__ == "__main__":
        a = 'train' #input('train or test:')
        train(a)
