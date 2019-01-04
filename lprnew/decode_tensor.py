#coding=utf-8
from collections import OrderedDict
import tensorflow as tf
from config_new import CHARS, CHARS_DICT

# def get_words_from_chars(characters_list: List[str], sequence_lengths: List[int], name='chars_conversion'):
#     with tf.name_scope(name=name):
#         def join_charcaters_fn(coords):
#             return tf.reduce_join(characters_list[coords[0]:coords[1]])
#
#         def coords_several_sequences():
#             end_coords = tf.cumsum(sequence_lengths)
#             start_coords = tf.concat([[0], end_coords[:-1]], axis=0)
#             coords = tf.stack([start_coords, end_coords], axis=1)
#             coords = tf.cast(coords, dtype=tf.int32)
#             return tf.map_fn(join_charcaters_fn, coords, dtype=tf.string)
#
#         def coords_single_sequence():
#             return tf.reduce_join(characters_list, keep_dims=True)
#
#         words = tf.cond(tf.shape(sequence_lengths)[0] > 1,
#                         true_fn=lambda: coords_several_sequences(),
#                         false_fn=lambda: coords_single_sequence())
#
#         return words


def decode_tensor(sparse_code):

    with tf.name_scope('code2str_conversion'):
        chars_dict_sorted = OrderedDict(sorted(CHARS_DICT.items(), key=lambda x: x[1]))
        chars_dict_sorted = {k.decode("utf-8"):v for k, v in chars_dict_sorted.iteritems()}

        codes = chars_dict_sorted.values()
        alphabet_units = chars_dict_sorted.keys()
        keys_alphabet_codes = tf.cast(codes, tf.int64)
        values_alphabet_units = [c for c in alphabet_units]
        table_int2str = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(keys_alphabet_codes, values_alphabet_units), '?')

        #sequence_lengths_pred = tf.bincount(tf.cast(sparse_code.indices[:, 0], tf.int32), minlength=1)

        pred_chars = table_int2str.lookup(sparse_code)
        #predicted = get_words_from_chars(pred_chars.values, sequence_lengths=sequence_lengths_pred)
        predicted = tf.reduce_join(pred_chars.values, keep_dims=True, name='predicted')

        return predicted