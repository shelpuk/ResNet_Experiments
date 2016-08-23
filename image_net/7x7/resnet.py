import numpy as np
import tensorflow as tf


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def softmax_layer(inpt, shape):
    fc_w = weight_variable(shape)
    fc_b = tf.Variable(tf.zeros([shape[1]]))

    logits = tf.matmul(inpt, fc_w) + fc_b
    activation = tf.nn.softmax(logits)

    return activation, logits

def conv_layer(inpt, filter_shape, stride):
    out_channels = filter_shape[3]

    filter_ = weight_variable(filter_shape)
    #print inpt.get_shape()
    #print filter_.get_shape()
    conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
    mean, var = tf.nn.moments(conv, axes=[0,1,2])
    beta = tf.Variable(tf.zeros([out_channels]), name="beta")
    gamma = weight_variable([out_channels], name="gamma")
    
    batch_norm = tf.nn.batch_norm_with_global_normalization(
        conv, mean, var, beta, gamma, 0.001,
        scale_after_normalization=True)

    out = tf.nn.relu(batch_norm)

    l2_loss = tf.nn.l2_loss(filter_)

    return out, l2_loss

def max_pool_layer(inpt, stride):
    filter_ = [1, stride, stride, 1]
    outpt = tf.nn.max_pool(inpt, ksize=filter_, strides=filter_, padding='SAME')
    return outpt

def residual_block(inpt, output_depth, down_sample, projection=False):
    input_depth = inpt.get_shape().as_list()[3]
    if down_sample:
        filter_ = [1,2,2,1]
        inpt = tf.nn.max_pool(inpt, ksize=filter_, strides=filter_, padding='SAME')

    [conv1, conv1_l2_loss] = conv_layer(inpt, [3, 3, input_depth, output_depth], 1)
    [conv2, conv2_l2_loss] = conv_layer(conv1, [3, 3, output_depth, output_depth], 1)

    if input_depth != output_depth:
        if projection:
            # Option B: Projection shortcut
            [input_layer, _] = conv_layer(inpt, [1, 1, input_depth, output_depth], 2)
        else:
            # Option A: Zero-padding
            input_layer = tf.pad(inpt, [[0,0], [0,0], [0,0], [0, output_depth - input_depth]])
    else:
        input_layer = inpt

    res = conv2 + input_layer
    sum_l2_loss = conv1_l2_loss + conv2_l2_loss
    return res, sum_l2_loss
