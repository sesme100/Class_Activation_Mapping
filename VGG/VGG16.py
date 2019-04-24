
import numpy as np
import tensorflow as tf

from Define import *

def Global_Average_Pooling(x, stride=1) :
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]

    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) 

def conv_relu(x, num_filter, training, layer_name):

    x = tf.layers.conv2d(inputs = x, filters = num_filter, kernel_size = [3, 3], strides = 1, padding = 'SAME', name = layer_name + '_conv_1')
    x = tf.nn.relu(x, name = layer_name + '_relu_1')
    return x

def VGG16(input, training_flag):
    x = input
    print(x)
    
    #block 1
    for i in range(2):
        x = conv_relu(x, CONV_FILTERS[0], training_flag, 'vgg_1_' + str(i))
        print(x)
    x = tf.layers.max_pooling2d(inputs = x, pool_size = [2, 2], strides = 2, name = 'vgg_1_pool_1')
    print(x)
    
    #block 2
    for i in range(2):
        x = conv_relu(x, CONV_FILTERS[1], training_flag, 'vgg_2_' + str(i))
        print(x)
    x = tf.layers.max_pooling2d(inputs = x, pool_size = [2, 2], strides = 2, name = 'vgg_2_pool_1')
    print(x)

    #block 3
    for i in range(3):
        x = conv_relu(x, CONV_FILTERS[2], training_flag, 'vgg_3_' + str(i))
        print(x)
    x = tf.layers.max_pooling2d(inputs = x, pool_size = [2, 2], strides = 2, name = 'vgg_3_pool_1')
    print(x)

    #block 4
    for i in range(3):
        x = conv_relu(x, CONV_FILTERS[3], training_flag, 'vgg_4_' + str(i))
        print(x)
    x = tf.layers.max_pooling2d(inputs = x, pool_size = [2, 2], strides = 2, name = 'vgg_4_pool_1')
    print(x)

    #block 5
    for i in range(3):
        x = conv_relu(x, CONV_FILTERS[3], training_flag, 'vgg_5_' + str(i))
        print(x)
    x = tf.layers.max_pooling2d(inputs = x, pool_size = [2, 2], strides = 2, name = 'vgg_5_pool_1')
    print(x) # (7, 7, 64)

    # GAP - ResNet, DenseNet
    x = Global_Average_Pooling(x) #(1, 1, 64)
    x = tf.contrib.layers.flatten(x) #(64)

    # FC - tf.layers.dense ( units = CLASSES)
    fc_w = tf.Variable(tf.random_normal([64, CLASSES]), name = 'fc_W')
    x = tf.matmul(x, fc_w, name = 'predict') # [CLASSES]

    return x

if __name__ == '__main__':
    input_var = tf.placeholder(tf.float32, [None, 32, 32, 3])
    output = VGG16(input_var, True)
    print(output)

'''
Tensor("Placeholder:0", shape=(?, 32, 32, 3), dtype=float32)
Tensor("vgg_1_0_relu_1:0", shape=(?, 32, 32, 8), dtype=float32)
Tensor("vgg_1_1_relu_1:0", shape=(?, 32, 32, 8), dtype=float32)
Tensor("vgg_1_pool_1/MaxPool:0", shape=(?, 16, 16, 8), dtype=float32)
Tensor("vgg_2_0_relu_1:0", shape=(?, 16, 16, 16), dtype=float32)
Tensor("vgg_2_1_relu_1:0", shape=(?, 16, 16, 16), dtype=float32)
Tensor("vgg_2_pool_1/MaxPool:0", shape=(?, 8, 8, 16), dtype=float32)
Tensor("vgg_3_0_relu_1:0", shape=(?, 8, 8, 32), dtype=float32)
Tensor("vgg_3_1_relu_1:0", shape=(?, 8, 8, 32), dtype=float32)
Tensor("vgg_3_2_relu_1:0", shape=(?, 8, 8, 32), dtype=float32)
Tensor("vgg_3_pool_1/MaxPool:0", shape=(?, 4, 4, 32), dtype=float32)
Tensor("vgg_4_0_relu_1:0", shape=(?, 4, 4, 64), dtype=float32)
Tensor("vgg_4_1_relu_1:0", shape=(?, 4, 4, 64), dtype=float32)
Tensor("vgg_4_2_relu_1:0", shape=(?, 4, 4, 64), dtype=float32)
Tensor("vgg_4_pool_1/MaxPool:0", shape=(?, 2, 2, 64), dtype=float32)
Tensor("vgg_5_0_relu_1:0", shape=(?, 2, 2, 64), dtype=float32)
Tensor("vgg_5_1_relu_1:0", shape=(?, 2, 2, 64), dtype=float32)
Tensor("vgg_5_2_relu_1:0", shape=(?, 2, 2, 64), dtype=float32)
Tensor("vgg_5_pool_1/MaxPool:0", shape=(?, 1, 1, 64), dtype=float32)
Tensor("predict:0", shape=(?, 10), dtype=float32)
'''

