import tensorflow as tf
from resnet import softmax_layer, conv_layer, residual_block, max_pool_layer

def resnet(inpt):
    layers = []
    total_l2_loss = 0

    with tf.variable_scope('conv1'):
        [conv1, conv1_l2_loss] = conv_layer(inpt, [7, 7, 3, 64], 2)
        conv1_pooled = max_pool_layer(conv1, 2)
        layers.append(conv1_pooled)
        total_l2_loss += conv1_l2_loss

    num_64_blocks = 3
    for i in range (num_64_blocks):
        with tf.variable_scope('conv2_%d' % (i+1)):
            [conv64, conv64_loss] = residual_block(layers[-1], 64, False)
            layers.append(conv64)
            total_l2_loss += conv64_loss

        assert conv64.get_shape().as_list()[1:] == [56, 56, 64]

    num_128_blocks = 4
    for i in range (num_128_blocks):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv3_%d' % (i+1)):
            [conv128, conv128_loss] = residual_block(layers[-1], 128, down_sample)
            layers.append(conv128)
            total_l2_loss += conv128_loss

        assert conv128.get_shape().as_list()[1:] == [28, 28, 128]

    num_256_blocks = 6
    for i in range (num_256_blocks):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv4_%d' % (i+1)):
            [conv256, conv256_loss] = residual_block(layers[-1], 256, down_sample)
            layers.append(conv256)
            total_l2_loss += conv256_loss

        assert conv256.get_shape().as_list()[1:] == [14, 14, 256]

    num_512_blocks = 3
    for i in range (num_512_blocks):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv5_%d' % (i+1)):
            [conv512, conv512_loss] = residual_block(layers[-1], 512, down_sample)
            layers.append(conv512)
            total_l2_loss += conv512_loss

        assert conv512.get_shape().as_list()[1:] == [7, 7, 512]

    with tf.variable_scope('fc'):
        global_pool = tf.reduce_mean(layers[-1], [1, 2])
        assert global_pool.get_shape().as_list()[1:] == [512]
        
        activation, logits = softmax_layer(global_pool, [512, 1000])
        layers.append(activation)

    return activation, logits, total_l2_loss / (len(layers) - 1)
