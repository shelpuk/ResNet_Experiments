import tensorflow as tf
from resnet import softmax_layer, conv_layer, residual_block

n_dict = {20:1, 32:2, 44:3, 56:4}
# ResNet architectures used for CIFAR-10
def resnet(inpt, n=32):
    if n < 20 or (n - 20) % 12 != 0:
        print "ResNet depth invalid."
        return

    num_conv = (n - 20) / 12 + 1
    layers = []
    total_l2_loss = 0

    with tf.variable_scope('conv1'):
        [conv1, conv1_l2_loss] = conv_layer(inpt, [3, 3, 3, 16], 1)
        layers.append(conv1)
        total_l2_loss += conv1_l2_loss

    for i in range (num_conv):
        with tf.variable_scope('conv2_%d' % (i+1)):
            [conv2_x, conv2_x_l2_loss] = residual_block(layers[-1], 16, False)
            [conv2, conv2_l2_loss] = residual_block(conv2_x, 16, False)
            layers.append(conv2_x)
            layers.append(conv2)
            total_l2_loss += conv2_x_l2_loss + conv2_l2_loss

        assert conv2.get_shape().as_list()[1:] == [32, 32, 16]

    for i in range (num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv3_%d' % (i+1)):
            [conv3_x, conv3_x_l2_loss] = residual_block(layers[-1], 32, down_sample)
            [conv3, conv3_l2_loss] = residual_block(conv3_x, 32, False)
            layers.append(conv3_x)
            layers.append(conv3)
            total_l2_loss += conv3_x_l2_loss + conv3_l2_loss

        assert conv3.get_shape().as_list()[1:] == [16, 16, 32]
    
    for i in range (num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv4_%d' % (i+1)):
            [conv4_x, conv4_x_l2_loss] = residual_block(layers[-1], 64, down_sample)
            [conv4, conv4_l2_loss] = residual_block(conv4_x, 64, False)
            layers.append(conv4_x)
            layers.append(conv4)
            total_l2_loss += conv4_x_l2_loss + conv4_l2_loss

        assert conv4.get_shape().as_list()[1:] == [8, 8, 64]

    with tf.variable_scope('fc'):
        global_pool = tf.reduce_mean(layers[-1], [1, 2])
        assert global_pool.get_shape().as_list()[1:] == [64]
        
        activation, logits = softmax_layer(global_pool, [64, 10])
        layers.append(activation)

    return activation, logits, total_l2_loss
