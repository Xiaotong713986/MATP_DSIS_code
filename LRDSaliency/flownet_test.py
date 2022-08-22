import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
import itertools
#from keras.utils.visualize_util import plot
import random
from scipy import misc
from scipy.linalg import logm, expm
import pandas as pd
import scipy
from os import listdir
from os.path import isfile, join
import matplotlib
# import corr_tf.
HEADLESS = False
if HEADLESS:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Use the custom correlation layer or build one from tensorflow slice operations
use_custom_correlation = True
if use_custom_correlation:
  import correlation_layer as cl

QUICK_DEBUG = True
BATCH_SIZE = 3
num_epochs = 10
num_train_sets = 9
loss_order = 2
batch_size = BATCH_SIZE

def myDot():
    return tf.keras.layers.Lambda(lambda x: tf.reduce_sum(tf.multiply(x[0],x[1]),axis=-1,keepdims=True),name = 'myDot')

def get_padded_stride(b,displacement_x,displacement_y,height_8=112/8,width_8=112/8):
    slice_height = height_8- abs(displacement_y)
    slice_width = width_8 - abs(displacement_x)
    start_y = abs(displacement_y) if displacement_y < 0 else 0
    start_x = abs(displacement_x) if displacement_x < 0 else 0
    top_pad    = displacement_y if (displacement_y>0) else 0
    bottom_pad = start_y
    left_pad   = displacement_x if (displacement_x>0) else 0
    right_pad  = start_x
    print top_pad, bottom_pad, left_pad, right_pad
    gather_layer = tf.keras.layers.Lambda(lambda x: tf.pad(tf.slice(x,begin=[0,start_y,start_x,0],size=[-1,slice_height,slice_width,-1]),paddings=[[0,0],[top_pad,bottom_pad],[left_pad,right_pad],[0,0]]),name='gather_{}_{}'.format(displacement_x,displacement_y))(b)
    return gather_layer

def get_correlation_layer(conv3_pool_l,conv3_pool_r,max_displacement=20,stride1=1, stride2=2,height_8=112/8,width_8=112/8):
    layer_list = []
    dotLayer = myDot()
    for i in range(-max_displacement, max_displacement+stride2,stride2):
        for j in range(-max_displacement, max_displacement+stride2,stride2):
            print conv3_pool_r,i,j,height_8,width_8
            slice_b = get_padded_stride(conv3_pool_r,i,j,height_8,width_8)
            current_layer = dotLayer([conv3_pool_l,slice_b])
            layer_list.append(current_layer)
    return tf.keras.layers.Lambda(lambda x: tf.concat(x, 3),name='441_output_concatenation')(layer_list)
    
def Intrinsic_Depth_Net(left, right):
    with tf.variable_scope('flownet-lr'):

        conv1a = tf.layers.conv2d(left, 64, [7, 7], [2, 2], 'same', name='conv1a')
        conv1b = tf.layers.conv2d(right, 64, [7, 7], [2, 2], 'same', name='conv1b')
        conv1a = tf.nn.leaky_relu(conv1a, 0.1, name='conv1a_relu')
        conv1b = tf.nn.leaky_relu(conv1b, 0.1, name='conv1b_relu')

        conv2a = tf.layers.conv2d(conv1a, 128, 5, 2, 'same', name='conv2a')
        conv2b = tf.layers.conv2d(conv1b, 128, 5, 2, 'same', name='conv2b')
        conv2a = tf.nn.leaky_relu(conv2a, 0.1, name='conv2a_relu')
        conv2b = tf.nn.leaky_relu(conv2b, 0.1, name='conv2b_relu')

        conv3a = tf.layers.conv2d(conv2a, 256, 5, 2, 'same', name='conv3a')
        conv3b = tf.layers.conv2d(conv2b, 256, 5, 2, 'same', name='conv3b')
        conv3a = tf.nn.leaky_relu(conv3a, 0.1, name='conv3a_relu')
        conv3b = tf.nn.leaky_relu(conv3b, 0.1, name='conv3b_relu')
        # merge
        print "Generating Correlation layer..."
        if use_custom_correlation:
            corr_layer = tf.keras.layers.Lambda( lambda x: cl.corr(a=x[0],b=x[1],stride=2,max_displacement=20), name= "correlation_layer")([conv3a,conv3b])
        else:
            corr_layer = get_correlation_layer(conv3a, conv3b, max_displacement=20,stride2=2,height_8=112/8,width_8=112/8)
        # print corr_layer
        corr_layer = tf.nn.leaky_relu(corr_layer, 0.1, name="corr_layer_relu")
        conv_redir = tf.layers.conv2d(conv3a, 32, 1, 1, 'valid', name='conv_redir')
        conv_redir =  tf.nn.leaky_relu(conv_redir, 0.1, name='conv_redir_relu')
        concatenator = tf.concat([corr_layer, conv_redir], axis=-1, name='concatenated_correlation')

        conv3_1 = tf.layers.conv2d(concatenator, 256, 3, 1, 'same', name='conv3_1')
        conv3_1 = tf.nn.leaky_relu(conv3_1, 0.1, name='conv3_1_relu')

        conv4 = tf.layers.conv2d(conv3_1, 512, 3, 2, 'same', name='conv4')
        conv4 = tf.nn.leaky_relu(conv4, 0.1, name='conv4_relu')

        conv4_1 = tf.layers.conv2d(conv4, 512, 3, 1, 'same', name='conv4_1')
        conv4_1 = tf.nn.leaky_relu(conv4_1, 0.1, name='conv4_1_relu')

        conv5 = tf.layers.conv2d(conv4_1, 512, 3, 2, 'same', name='conv5')
        conv5 = tf.nn.leaky_relu(conv5, 0.1, name='conv5_relu')

        conv5_1 = tf.layers.conv2d(conv5, 512, 3, 1, 'same', name='conv5_1')
        conv5_1 = tf.nn.leaky_relu(conv5_1, 0.1, name='conv5_1_relu')

        conv6 = tf.layers.conv2d(conv5_1, 1024, 3, 2, 'same', name='conv6')
        conv6 = tf.nn.leaky_relu(conv6, 0.1, name='conv6_relu')
        # print conv6
        conv6_1 = tf.layers.conv2d(conv6, 1024, 3, 1, 'same', name='conv6_1')
        conv6_1 = tf.nn.leaky_relu(conv6_1, 0.1, name='conv6_1_relu')
        print conv6_1
        print "Upsampling..."
        # upsampling layers
        
        deconv_1_a_kernel = tf.constant(1.0, shape=[8,8,512,1024])
        deconv1_a = tf.nn.conv2d_transpose(value=conv6_1, filter=deconv_1_a_kernel, output_shape=[1, 10, 10, 512], strides=[1, 2, 2, 1], padding= 'SAME', name='deconv1_a')
        deconv1_a = tf.layers.batch_normalization(deconv1_a, name='deconv1_a_bn')
        deconv1_a = tf.nn.relu(deconv1_a, name='deconv1_a_relu')
        deconv1_a = tf.layers.dropout(deconv1_a, rate=0.2)
        print deconv1_a
        deconv_2_a_kernel = tf.constant(1.0, shape=[4,4,128,512])
        deconv2_a = tf.nn.conv2d_transpose(value=deconv1_a, filter=deconv_2_a_kernel, output_shape=[1, 28, 28, 128], strides=[1, 4, 4, 1], padding='SAME', name='deconv2_a')
        deconv2_a = tf.layers.batch_normalization(deconv2_a, name='deconv2_a_bn')
        deconv2_a = tf.nn.relu(deconv2_a, name='deconv2_a_relu')
        deconv2_a = tf.layers.dropout(deconv2_a, rate=0.2)
        print deconv2_a
        deconv_3_a_kernel = tf.constant(1.0, shape=[4,4,32,128])
        deconv3_a = tf.nn.conv2d_transpose(value=deconv2_a, filter=deconv_3_a_kernel, output_shape=[1, 56, 56, 32], strides=[1, 2, 2, 1], padding='SAME', name='deconv3_a')
        deconv3_a = tf.layers.batch_normalization(deconv3_a, name='deconv3_a_bn')
        deconv3_a = tf.nn.relu(deconv3_a, name='deconv3_a_relu')
        deconv3_a = tf.layers.dropout(deconv3_a, rate=0.2)
        print deconv3_a
        predict_kernel = tf.constant(1.0, shape=[4,4,1,32])
        predict = tf.nn.conv2d_transpose(value=deconv3_a, filter=predict_kernel, output_shape=[1, 112, 112, 1], strides=[1, 2, 2, 1], padding='SAME', name='predict')
        # loss SigmoidCrossEntropyLoss
        print predict
        
        print "Done"

if __name__ == '__main__':
    with tf.Session() as sess:
        left = tf.get_variable('left', shape=[1, 112, 112, 3])
        right = tf.get_variable('right', shape=[1, 112, 112, 3])
        print ("variable created..")
        model = Intrinsic_Depth_Net(left, right)
