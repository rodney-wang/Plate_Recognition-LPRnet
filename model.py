#coding=utf-8
import tensorflow as tf
from config_new import NUM_CHARS

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

def get_train_model(num_channels, label_len, b, img_size):
    inputs = tf.placeholder(
        tf.float32,
        shape=(b, img_size[0], img_size[1], num_channels))

    # 定义ctc_loss需要的稀疏矩阵
    targets = tf.sparse_placeholder(tf.int32)

    # 1维向量 序列长度 [batch_size,]
    seq_len = tf.placeholder(tf.int32, [None])
    x = inputs

    x = conv(x,num_channels,64,ksize=[3,3])
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.nn.max_pool(x,
                          ksize=[1, 3, 3, 1],
                          strides=[1, 1, 1, 1],
                          padding='SAME')
    x = small_basic_block(x,64,64)
    x2=x
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)

    x = tf.nn.max_pool(x,
                          ksize=[1, 3, 3, 1],
                          strides=[1, 2, 1, 1],
                          padding='SAME')
    x = small_basic_block(x, 64,256)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = small_basic_block(x, 256, 256)
    x3 = x
    x = tf.layers.batch_normalization(x)

    x = tf.nn.relu(x)
    x = tf.nn.max_pool(x,
                       ksize=[1, 3, 3, 1],
                       strides=[1, 2, 1, 1],
                       padding='SAME')
    x = tf.layers.dropout(x)

    x = conv(x, 256, 256, ksize=[4, 1])
    x = tf.layers.dropout(x)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)


    x = conv(x,256,NUM_CHARS+1,ksize=[1,13],pad='SAME')
    x = tf.nn.relu(x)
    cx = tf.reduce_mean(tf.square(x))
    x = tf.div(x,cx)

    #x = tf.reduce_mean(x,axis = 2)
    #x1 = conv(inputs,num_channels,num_channels,ksize = (5,1))


    x1 = tf.nn.avg_pool(inputs,
                       ksize=[1, 4, 1, 1],
                       strides=[1, 4, 1, 1],
                       padding='SAME')
    cx1 = tf.reduce_mean(tf.square(x1))
    x1 = tf.div(x1, cx1)

    # x1 = tf.image.resize_images(x1, size = [18, 16], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    x2 = tf.nn.avg_pool(x2,
                        ksize=[1, 4, 1, 1],
                        strides=[1, 4, 1, 1],
                        padding='SAME')
    cx2 = tf.reduce_mean(tf.square(x2))
    x2 = tf.div(x2, cx2)

    #x2 = tf.image.resize_images(x2, size=[18, 16], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    x3 = tf.nn.avg_pool(x3,
                        ksize=[1, 2, 1, 1],
                        strides=[1, 2, 1, 1],
                        padding='SAME')
    cx3 = tf.reduce_mean(tf.square(x3))
    x3 = tf.div(x3, cx3)

    #x3 = tf.image.resize_images(x3, size=[18, 16], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


    #x1 = tf.nn.relu(x1)

    x = tf.concat([x,x1,x2,x3],3)
    x = conv(x, x.get_shape().as_list()[3], NUM_CHARS + 1, ksize=(1, 1))
    logits = tf.reduce_mean(x,axis=2)
    # x_shape = x.get_shape().as_list()
    # outputs = tf.reshape(x, [-1,x_shape[2]*x_shape[3]])
    # W1 = tf.Variable(tf.truncated_normal([x_shape[2]*x_shape[3],
    #                                      150],
    #                                     stddev=0.1))
    # b1 = tf.Variable(tf.constant(0., shape=[150]))
    # # [batch_size*max_timesteps,num_classes]
    # x = tf.matmul(outputs, W1) + b1
    # x= tf.layers.dropout(x)
    # x = tf.nn.relu(x)
    # W2 = tf.Variable(tf.truncated_normal([150,
    #                                      NUM_CHARS+1],
    #                                     stddev=0.1))
    # b2 = tf.Variable(tf.constant(0., shape=[NUM_CHARS+1]))
    # x = tf.matmul(x, W2) + b2
    # x = tf.layers.dropout(x)
    # # [batch_size,max_timesteps,num_classes]
    # logits = tf.reshape(x, [b, -1, NUM_CHARS+1])

    return logits, inputs, targets, seq_len
