import os
import argparse
import tensorflow as tf
import numpy as np
from model import get_train_model
from config import img_size, num_channels, label_len, NUM_CHARS
from decode_tensor import decode_tensor

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

"""
Inputs to the LPRNet: 
 - Inputs:   images with shape [BATCH_SIZE, img_size[0], img_size[1], 3]
 - seq_len:  np.ones(BATCH_SIZE) * 24 
 for more details, please check "model.py" 

Outputs: 
 - decoded:  carplates decoded as sequences of numbers, each number represent a character
 - score  :  confidence score of the detection

"""

BATCH_SIZE = 1
test_seq_len = tf.ones(BATCH_SIZE) * 24

parser = argparse.ArgumentParser(description='Plate end to end test')
parser.add_argument('--model_ckpt', default='/ssd//wfei/code/Plate_Recognition-LPRnet/model_h5/LPRc1.ckpt-63000',
                        type=str, help='Path to model checkpoint')
parser.add_argument('--model_pb', default='./model_pb',
                        type=str, help='Output model pb path')
parser.add_argument('--num_channels', default=1,
                        type=str, help='Number of channels')
args = parser.parse_args()


MODEL_DIR = './model_pb/model_k11_pb_synthetic'
MODEL_CKPT = './model_wanda/LPR_wanda.ckpt-100000'
MODEL_CKPT = './model_wanda_fresh_0604/LPR_wanda.ckpt-150000' 
MODEL_CKPT = './model_k11/LPR_k11_withsyn.ckpt-70000'
args.num_channels=1

tf.reset_default_graph()
eval_graph = tf.Graph()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
exporter = tf.saved_model.builder.SavedModelBuilder(MODEL_DIR)

with tf.Session(config=config, graph=eval_graph) as sess:

    global_step = tf.Variable(0, trainable=False)

    #isTraining = tf.placeholder(tf.bool, name="is_train")
    logits, inputs, targets, seq_len = get_train_model(args.num_channels, label_len, BATCH_SIZE, img_size, False, False)

    logits = tf.transpose(logits, (1, 0, 2), name='logits_transpose')

    #decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False, beam_width=100, top_paths=3)
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False, top_paths=3)
    plate_predict = decode_tensor(decoded[0])
    score = tf.subtract(log_prob[:, 0], log_prob[:, 1], name='confidence_score')
    # feed_dict = {"inputs": inputs,
    #             "seq_len": test_seq_len}
    ipt = {"Placeholder": inputs,
           "Placeholder_4":seq_len}
           #"is_train": isTraining}

    outputs = { 'code2str_conversion/predicted':plate_predict,
                'confidence_score': score}

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=4)
    print tf.global_variables()
    #saver = tf.train.import_meta_graph(MODEL_CKPT + '.meta', clear_devices=True)
    saver.restore(sess, MODEL_CKPT)

    exporter.add_meta_graph_and_variables(
        sess,
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                tf.saved_model.signature_def_utils.predict_signature_def(inputs=ipt,
                                                                         outputs=outputs)
        },
        legacy_init_op=tf.tables_initializer()
    )
    exporter.save()
