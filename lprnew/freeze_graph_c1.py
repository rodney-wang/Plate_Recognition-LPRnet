import os, argparse

import tensorflow as tf
from model import get_train_model
from config_new import img_size, label_len, NUM_CHARS
from decode_tensor import decode_tensor

# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph

dir = os.path.dirname(os.path.realpath(__file__))
BATCH_SIZE = 1
test_seq_len = tf.ones(BATCH_SIZE) * 24

def freeze_graph(model_dir, output_node_names):
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names,
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = './modelk11/LPRChar69.ckpt-96000'
    input_checkpoint = './model_c1/LPRc1.ckpt-63000'

    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    print absolute_model_dir
    output_graph = absolute_model_dir + "/frozen_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True
    tf.reset_default_graph()
    eval_graph = tf.Graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # We start a session using a temporary fresh Graph
    #with tf.Session(graph=tf.Graph()) as sess:
    with tf.Session(config=config, graph=eval_graph) as sess:
        global_step = tf.Variable(0, trainable=False)

        # We import the meta graph in the current default Graph
        #saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # isTraining = tf.placeholder(tf.bool, name="is_train")
        logits, inputs, targets, seq_len = get_train_model(1, label_len, BATCH_SIZE, img_size, False,
                                                           False)

        logits = tf.transpose(logits, (1, 0, 2), name='logits_transpose')

        # decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False, beam_width=100, top_paths=3)
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False, top_paths=3)
        plate_predict = decode_tensor(decoded[0])
        score = tf.subtract(log_prob[:, 0], log_prob[:, 1], name='confidence_score')

        # We restore the weights
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=4)
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument("--model_dir", type=str, default="./modelk11", help="Model folder to export")
    parser.add_argument("--model_dir", type=str, default="./model_c1", help="Model folder to export")
    parser.add_argument("--output_node_names", type=str, default="code2str_conversion/predicted,confidence_score",
                        help="The name of the output nodes, comma separated.")
    args = parser.parse_args()

    freeze_graph(args.model_dir, args.output_node_names)