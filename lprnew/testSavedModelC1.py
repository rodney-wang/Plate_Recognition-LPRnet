import cv2
import tensorflow as tf
import numpy as np
import argparse


def print_all_variables():

    # Print all operators in the graph
    for op in sess.graph.get_operations():
        print(op)
    # Print all tensors produced by each operator in the graph
    print "\n\n####### OP VALUES ###########"
    #for op in sess.graph.get_operations():
    #    print(op.values())
    tensor_names = [[v.name for v in op.values()] for op in sess.graph.get_operations()]
    tensor_names = np.squeeze(tensor_names)
    print "\n\n#######TENSOR NAMES###########"
    print(tensor_names)

parser = argparse.ArgumentParser(description='Plate end to end test')
parser.add_argument('--img', default='/ssd/wfei/data/testing_data/k11_plates_v1.2/ch44009_20181014143821-00000070-0118.jpg_plate.png',
                        type=str, help='Input test image dir')

args = parser.parse_args()


fname = '/Users/fei/tmp/plate.jpg'
#fname = '/Users/fei/data/parking/carplate/testing_data/k11_benchmark/k11_plates_v1.2/ch44009_20181014143821-00000070-0118.jpg_plate.png'
#fname = '/Users/fei/data/parking/carplate/forlabel/energy_cars_plates/1540553852598657412_plate.jpg'
#img = cv2.imread(args.img)
img = cv2.imread(fname)
print img.shape
img = cv2.resize(img, (94, 24), interpolation=cv2.INTER_CUBIC)
images = img[np.newaxis, :]
images = np.transpose(images, axes=[0, 2, 1, 3])


with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], './model_pb')
    graph = tf.get_default_graph()

    #print_all_variables()

    codes = graph.get_tensor_by_name("CTCBeamSearchDecoder:3")
    decoded = graph.get_tensor_by_name("code2str_conversion/predicted:0")
    score = graph.get_tensor_by_name("confidence_score:0")


    inputs = {"Placeholder:0": images,
             "Placeholder_4:0": [24],
              "is_train:0":False}


    print "\n\n###### OUTPUT #####"
    #print(sess.run([codes, score], inputs))
    codes_pred, chars_pred, score_pred = sess.run([codes, decoded, score], inputs)
    #print(sess.run([codes, decoded, score], inputs))
    print codes_pred, chars_pred, score_pred
    print chars_pred[0].decode('utf8')

