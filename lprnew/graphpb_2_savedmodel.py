import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

export_dir = './saved'
graph_pb = './modelk11/frozen_model.pb'

builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

with tf.gfile.GFile(graph_pb, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

sigs = {}

with tf.Session(graph=tf.Graph()) as sess:
    # name="" is important to ensure we don't get spurious prefixing
    tf.import_graph_def(graph_def, name="")
    g = tf.get_default_graph()

    inp     = g.get_tensor_by_name("Placeholder:0")
    seq_len = g.get_tensor_by_name("Placeholder_4:0")
    istrain = g.get_tensor_by_name("is_train:0")

    #outputs = { 'code2str_conversion/predicted:0':plate_predict,
    #            'confidence_score': score}

    predict = g.get_tensor_by_name("code2str_conversion/predicted:0")
    score   = g.get_tensor_by_name("confidence_score:0")


    sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
        tf.saved_model.signature_def_utils.predict_signature_def(
            {"Placeholder": inp, "Placeholder_4": seq_len, "isTraining":istrain},
            {"code2str_conversion/predicted": predict, "confidence_score":score})

    builder.add_meta_graph_and_variables(sess,
                                         [tag_constants.SERVING],
                                         signature_def_map=sigs,
                                        legacy_init_op = tf.tables_initializer()
    )

builder.save()