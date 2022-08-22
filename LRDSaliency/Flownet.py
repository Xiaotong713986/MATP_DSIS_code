# encoding: utf-8
'''
    Author: leesky
    Date: 2018/12/20
'''
import cv2
import tensorflow as tf
import sys, os, h5py
import numpy as np

from utils1 import LeakyReLU, average_endpoint_error, pad, antipad, crop_features
import tensorflow as tf
slim = tf.contrib.slim

# from ops import *

HEADLESS = False
if HEADLESS:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Use the custom correlation layer or build one from tensorflow slice operations
use_custom_correlation = True
if use_custom_correlation:
  import correlation_layer as cl

BATCH_SIZE = 1
CROP_HEIGHT = 112
CROP_WIDTH = 112


class Flownet():
    def __init__(self, sess, learning_rate=1e-2, batch_size=1):
        self.sess = sess
        # self.name = "Flownet"
        # self.dataset_name = args.dataset

        # self.checkpoint_dir = args.checkpoint_dir
        # self.sample_dir = args.sample_dir
        # self.result_dir = args.result_dir
        # self.log_dir = args.log_dir

        # self.epoch = args.epoch
        # self.iteration = args.iteration
        # self.batch_size = args.batch_size
        self.batch_size = batch_size
        # self.print_freq = args.print_freq
        # self.save_freq = args.save_freq
        # self.img_size = args.img_size
        self.img_size = 112
        self.c_dim = 3
        self.training = True
        # train
        # self.learning_rate = args.lr
        self.base_lr = learning_rate
        self.decay_rate = 0.1
        self.decay_steps = 20000
        # self.beta = args.beta


        self.custom_dataset = True
        # if self.dataset_name = 'svsd' :
        #     self.c_dim = 3
        #     self.data = load_data(dataset_name=self.dataset_name, size=self.img_size)

        print()

        print("##### Information #####")
        print("# batch_size : ", self.batch_size)
        # print("# epoch : ", self.epoch)
        # print("# iteration per epoch : ", self.iteration)


    ##################################################################################
    # utils for flownet
    ##################################################################################

    def get_correlation_layer(self, conv3_pool_l,conv3_pool_r,max_displacement=20, stride2=2,height_8=14,width_8=14):
        layer_list = []
        dotLayer = self.myDot()
        for i in range(-max_displacement, max_displacement+stride2,stride2):
            for j in range(-max_displacement, max_displacement+stride2,stride2):
                slice_b = self.get_padded_stride(conv3_pool_r,i,j,height_8,width_8)
                current_layer = dotLayer([conv3_pool_l,slice_b])
                layer_list.append(current_layer)
        return tf.keras.layers.Lambda(lambda x: tf.concat(x, 3),name='441_output_concatenation')(layer_list)
        
    def myDot(self):
        return tf.keras.layers.Lambda(lambda x: tf.reduce_sum(tf.multiply(x[0],x[1]),axis=-1,keepdims=True),name = 'myDot')

    def get_padded_stride(self, b,displacement_x,displacement_y,height_8=14,width_8=14):
        slice_height = height_8- abs(displacement_y)
        slice_width = width_8 - abs(displacement_x)
        start_y = abs(displacement_y) if displacement_y < 0 else 0
        start_x = abs(displacement_x) if displacement_x < 0 else 0
        top_pad    = displacement_y if (displacement_y>0) else 0
        bottom_pad = start_y
        left_pad   = displacement_x if (displacement_x>0) else 0
        right_pad  = start_x
        
        gather_layer = tf.keras.layers.Lambda(lambda x: tf.pad(tf.slice(x,begin=[0,start_y,start_x,0],size=[-1,slice_height,slice_width,-1]),paddings=[[0,0],[top_pad,bottom_pad],[left_pad,right_pad],[0,0]]),name='gather_{}_{}'.format(displacement_x,displacement_y))(b)
        return gather_layer

    ##################################################################################
    # Flownet
    ##################################################################################

    def weight_variable(self, shape):
        initializer=tf.contrib.layers.variance_scaling_initializer()
        return tf.Variable(initial)
    
    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial)

    def model(self, left, right, is_training):
        with tf.variable_scope("flownet", reuse=tf.AUTO_REUSE):
            conv1a = tf.layers.conv2d(left, 64, [7, 7], [2, 2], 'same', name='conv1a', kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            conv1b = tf.layers.conv2d(right, 64, [7, 7], [2, 2], 'same', name='conv1b', kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            conv1a = tf.nn.leaky_relu(conv1a, 0.1, name='conv1a_relu')
            conv1b = tf.nn.leaky_relu(conv1b, 0.1, name='conv1b_relu')

            conv2a = tf.layers.conv2d(conv1a, 128, 5, 2, 'same', name='conv2a', kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            conv2b = tf.layers.conv2d(conv1b, 128, 5, 2, 'same', name='conv2b', kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            conv2a = tf.nn.leaky_relu(conv2a, 0.1, name='conv2a_relu')
            conv2b = tf.nn.leaky_relu(conv2b, 0.1, name='conv2b_relu')

            conv3a = tf.layers.conv2d(conv2a, 256, 5, 2, 'same', name='conv3a', kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            conv3b = tf.layers.conv2d(conv2b, 256, 5, 2, 'same', name='conv3b', kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            conv3a = tf.nn.leaky_relu(conv3a, 0.1, name='conv3a_relu')
            conv3b = tf.nn.leaky_relu(conv3b, 0.1, name='conv3b_relu')
            # merge

            if use_custom_correlation:
                corr_layer = tf.keras.layers.Lambda( lambda x: cl.corr(a=x[0],b=x[1],stride=2,max_displacement=20), name= "correlation_layer")([conv3a,conv3b])
            else:
                corr_layer = self.get_correlation_layer(conv3a, conv3b, max_displacement=20,stride2=2,height_8=14,width_8=14)
            # print corr_layer
            corr_layer = tf.nn.leaky_relu(corr_layer, 0.1, name="corr_layer_relu")
            conv_redir = tf.layers.conv2d(conv3a, 32, 1, 1, 'valid', name='conv_redir', kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            conv_redir =  tf.nn.leaky_relu(conv_redir, 0.1, name='conv_redir_relu')
            concatenator = tf.concat([corr_layer, conv_redir], axis=-1, name='concatenated_correlation')

            conv3_1 = tf.layers.conv2d(concatenator, 256, 3, 1, 'same', name='conv3_1', kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            conv3_1 = tf.nn.leaky_relu(conv3_1, 0.1, name='conv3_1_relu')

            conv4 = tf.layers.conv2d(conv3_1, 512, 3, 2, 'same', name='conv4', kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            conv4 = tf.nn.leaky_relu(conv4, 0.1, name='conv4_relu')

            conv4_1 = tf.layers.conv2d(conv4, 512, 3, 1, 'same', name='conv4_1', kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            conv4_1 = tf.nn.leaky_relu(conv4_1, 0.1, name='conv4_1_relu')

            conv5 = tf.layers.conv2d(conv4_1, 512, 3, 2, 'same', name='conv5', kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            conv5 = tf.nn.leaky_relu(conv5, 0.1, name='conv5_relu')

            conv5_1 = tf.layers.conv2d(conv5, 512, 3, 1, 'same', name='conv5_1', kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            conv5_1 = tf.nn.leaky_relu(conv5_1, 0.1, name='conv5_1_relu')

            conv6 = tf.layers.conv2d(conv5_1, 1024, 3, 2, 'same', name='conv6', kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            conv6 = tf.nn.leaky_relu(conv6, 0.1, name='conv6_relu')
            # print conv6
            conv6_1 = tf.layers.conv2d(conv6, 1024, 3, 1, 'same', name='conv6_1', kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            conv6_1 = tf.nn.leaky_relu(conv6_1, 0.1, name='conv6_1_relu')
            # print conv6_1
            # upsampling layers
            deconv1_a = tf.layers.conv2d_transpose(conv6_1, 512, 8, 4, padding='SAME', name='deconv1_a', kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            deconv1_a = tf.layers.batch_normalization(deconv1_a, name='deconv1_a_bn', training=is_training)
            deconv1_a = tf.nn.relu(deconv1_a, name='deconv1_a_relu')
            deconv1_a = tf.layers.dropout(deconv1_a, rate=0.2, training=is_training)

            deconv2_a = tf.layers.conv2d_transpose(deconv1_a, 128, 7, 3, padding='VALID', name='deconv2_a', kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            deconv2_a = tf.layers.batch_normalization(deconv2_a, name='deconv2_a_bn', training=is_training)
            deconv2_a = tf.nn.relu(deconv2_a, name='deconv2_a_relu')
            deconv2_a = tf.layers.dropout(deconv2_a, rate=0.2, training=is_training)

            deconv3_a = tf.layers.conv2d_transpose(deconv2_a, 32, 4, 2, padding='SAME', name='deconv_3_a', kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            deconv3_a = tf.layers.batch_normalization(deconv3_a, name='deconv3_a_bn', training=is_training)
            deconv3_a = tf.nn.relu(deconv3_a, name='deconv3_a_relu')
            deconv3_a = tf.layers.dropout(deconv3_a, rate=0.2, training=is_training)

            predict = tf.layers.conv2d_transpose(deconv3_a, 1, 4, 2, padding='SAME', name='predict', kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            predict = tf.nn.relu(predict, name="predict_relu")
            return predict

    def fusenet(self, left, right, trainable=True):
        _, height, width, _ = left.shape.as_list()
        with tf.variable_scope('FlowNetC', reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                # Only backprop this network if trainable
                                trainable=trainable,
                                # He (aka MSRA) weight initialization
                                weights_initializer=slim.variance_scaling_initializer(),
                                activation_fn=LeakyReLU,
                                # We will do our own padding to match the original Caffe code
                                padding='VALID'):

                weights_regularizer = slim.l2_regularizer(0.00005)
                with slim.arg_scope([slim.conv2d], weights_regularizer=weights_regularizer):
                    with slim.arg_scope([slim.conv2d], stride=2):
                        conv_a_1 = slim.conv2d(pad(left, 3), 64, 7, scope='conv1')
                        conv_a_2 = slim.conv2d(pad(conv_a_1, 2), 128, 5, scope='conv2')
                        conv_a_3 = slim.conv2d(pad(conv_a_2, 2), 256, 5, scope='conv3')

                        conv_b_1 = slim.conv2d(pad(right, 3),
                                               64, 7, scope='conv1', reuse=True)
                        conv_b_2 = slim.conv2d(pad(conv_b_1, 2), 128, 5, scope='conv2', reuse=True)
                        conv_b_3 = slim.conv2d(pad(conv_b_2, 2), 256, 5, scope='conv3', reuse=True)

                        # Compute cross correlation with leaky relu activation
                        # cc = correlation(conv_a_3, conv_b_3, 1, 20, 1, 2, 20)
                        if use_custom_correlation:
                            cc = tf.keras.layers.Lambda( lambda x: cl.corr(a=x[0],b=x[1],stride=2,max_displacement=20), name= "correlation_layer")([conv_a_3,conv_b_3])
                        else:
                            cc = self.get_correlation_layer(conv3a, conv3b, max_displacement=20,stride2=2,height_8=14,width_8=14)
                        cc_relu = LeakyReLU(cc)

                    # Combine cross correlation results with convolution of feature map A
                    netA_conv = slim.conv2d(conv_a_3, 32, 1, scope='conv_redir')
                    # Concatenate along the channels axis
                    net = tf.concat([netA_conv, cc_relu], axis=3)

                    conv3_1 = slim.conv2d(pad(net), 256, 3, scope='conv3_1')
                    with slim.arg_scope([slim.conv2d], num_outputs=512, kernel_size=3):
                        conv4 = slim.conv2d(pad(conv3_1), stride=2, scope='conv4')
                        conv4_1 = slim.conv2d(pad(conv4), scope='conv4_1')
                        conv5 = slim.conv2d(pad(conv4_1), stride=2, scope='conv5')
                        conv5_1 = slim.conv2d(pad(conv5), scope='conv5_1')
                    conv6 = slim.conv2d(pad(conv5_1), 1024, 3, stride=2, scope='conv6')
                    conv6_1 = slim.conv2d(pad(conv6), 1024, 3, scope='conv6_1')
                    print conv6_1

                    # ''' START: upsampling '''
                    # with slim.arg_scope([slim.conv2d_transpose], biases_initializer=None):
                    #     deconv1_a = slim.conv2d_transpose(conv6_1, 512, 8, stride=4, padding='SAME', scope='deconv1')
                    #     deconv1_a = tf.layers.batch_normalization(deconv1_a, name='deconv1_a_bn')
                    #     deconv1_a = tf.nn.relu(deconv1_a, name='deconv1_a_relu')
                    #     deconv1_a = tf.layers.dropout(deconv1_a, rate=0.2, training=trainable)
                    #     print deconv1_a
                    #     deconv2_a = slim.conv2d_transpose(deconv1_a, 128, 7, 3, padding='VALID', name='deconv2_a', kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
                    #     deconv2_a = tf.layers.batch_normalization(deconv2_a, name='deconv2_a_bn')
                    #     deconv2_a = tf.nn.relu(deconv2_a, name='deconv2_a_relu')
                    #     deconv2_a = tf.layers.dropout(deconv2_a, rate=0.2, training=trainable)
                    #     print deconv2_a
                    #     deconv3_a = tf.layers.conv2d_transpose(deconv2_a, 32, 4, 2, padding='SAME', name='deconv_3_a', kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
                    #     deconv3_a = tf.layers.batch_normalization(deconv3_a, name='deconv3_a_bn')
                    #     deconv3_a = tf.nn.relu(deconv3_a, name='deconv3_a_relu')
                    #     deconv3_a = tf.layers.dropout(deconv3_a, rate=0.2, training=trainable)
                    #     print deconv3_a
                    #     predict = tf.layers.conv2d_transpose(deconv3_a, 1, 4, 2, padding='SAME', name='predict', kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

                    """ START: Refinement Network """
                    with slim.arg_scope([slim.conv2d_transpose], biases_initializer=None):
                        # H/64 x W/64
                        predict_flow6 = slim.conv2d(pad(conv6_1), 2, 3, scope='predict_flow6', activation_fn=None)
                        deconv5 = slim.conv2d_transpose(conv6_1, 512, 4, stride=2, scope='deconv5')
                        upsample_flow6to5 = slim.conv2d_transpose(predict_flow6, 2, 4, stride=2, scope='upsample_flow6to5', activation_fn=None)
                        deconv5 = crop_features(deconv5, tf.shape(conv5_1))
                        upsample_flow6to5 = crop_features(upsample_flow6to5, tf.shape(conv5_1))
                        concat5 = tf.concat([conv5_1, deconv5, upsample_flow6to5], axis=3)

                        # H/32 x W/32
                        predict_flow5 = slim.conv2d(pad(concat5), 2, 3, scope='predict_flow5', activation_fn=None)
                        deconv4 = slim.conv2d_transpose(concat5, 256, 4, stride=2, scope='deconv4')
                        upsample_flow5to4 = slim.conv2d_transpose(predict_flow5, 2, 4, stride=2, scope='upsample_flow5to4', activation_fn=None)
                        deconv4 = crop_features(deconv4, tf.shape(conv4_1))
                        upsample_flow5to4 = crop_features(upsample_flow5to4, tf.shape(conv4_1))
                        concat4 = tf.concat([conv4_1, deconv4, upsample_flow5to4], axis=3)

                        # H/16 x W/16
                        predict_flow4 = slim.conv2d(pad(concat4), 2, 3, scope='predict_flow4', activation_fn=None)
                        deconv3 = slim.conv2d_transpose(concat4, 128, 4, stride=2, scope='deconv3')
                        upsample_flow4to3 = slim.conv2d_transpose(predict_flow4, 2, 4, stride=2, scope='upsample_flow4to3', activation_fn=None)
                        deconv3 = crop_features(deconv3, tf.shape(conv3_1))
                        upsample_flow4to3 = crop_features(upsample_flow4to3, tf.shape(conv3_1))
                        concat3 = tf.concat([conv3_1, deconv3, upsample_flow4to3], axis=3)
                        print concat3
                        # H/8 x W/8
                        predict_flow3 = slim.conv2d(pad(concat3), 2, 3, scope='predict_flow3', activation_fn=None)
                        deconv2 = slim.conv2d_transpose(concat3, 64, 4, stride=2, scope='deconv2') 
                        upsample_flow3to2 = slim.conv2d_transpose(predict_flow3, 2, 4, stride=2, scope='upsample_flow3to2', activation_fn=None)
                        deconv2 = crop_features(deconv2, tf.shape(conv_a_2))
                        upsample_flow3to2 = crop_features(upsample_flow3to2, tf.shape(conv_a_2))
                        concat2 = tf.concat([conv_a_2, deconv2, upsample_flow3to2], axis=3)
                        print deconv2
                        predict_flow2 = slim.conv2d(pad(concat2), 1, 3, scope='predict_flow2', activation_fn=None)
                    """ END: Refinement Network """

                    flow = predict_flow2 * 20.0
                    # TODO: Look at Accum (train) or Resample (deploy) to see if we need to do something different
                    flow = tf.image.resize_bilinear(flow,
                                                    tf.stack([height, width]),
                                                    align_corners=True)
                    print flow
                    return flow
            


    def fusenet2(self, left, right, trainable=True):
        _, height, width, _ = left.shape.as_list()
        with tf.variable_scope('FlowNetC', reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                # Only backprop this network if trainable
                                # trainable=trainable,
                                # He (aka MSRA) weight initialization
                                weights_initializer=slim.variance_scaling_initializer(),
                                activation_fn=LeakyReLU,
                                # We will do our own padding to match the original Caffe code
                                padding='VALID'):

                weights_regularizer = slim.l2_regularizer(0.00005)
                with slim.arg_scope([slim.conv2d], weights_regularizer=weights_regularizer):
                    with slim.arg_scope([slim.conv2d], stride=2):
                        conv_a_1 = slim.conv2d(pad(left, 3), 64, 7, scope='conv1')
                        conv_a_2 = slim.conv2d(pad(conv_a_1, 2), 128, 5, scope='conv2')
                        conv_a_3 = slim.conv2d(pad(conv_a_2, 2), 256, 5, scope='conv3')

                        conv_b_1 = slim.conv2d(pad(right, 3),
                                               64, 7, scope='conv1', reuse=True)
                        conv_b_2 = slim.conv2d(pad(conv_b_1, 2), 128, 5, scope='conv2', reuse=True)
                        conv_b_3 = slim.conv2d(pad(conv_b_2, 2), 256, 5, scope='conv3', reuse=True)

                        # Compute cross correlation with leaky relu activation
                        # cc = correlation(conv_a_3, conv_b_3, 1, 20, 1, 2, 20)
                        if use_custom_correlation:
                            cc = tf.keras.layers.Lambda( lambda x: cl.corr(a=x[0],b=x[1],stride=2,max_displacement=20), name= "correlation_layer")([conv_a_3,conv_b_3])
                        else:
                            cc = self.get_correlation_layer(conv3a, conv3b, max_displacement=20,stride2=2,height_8=14,width_8=14)
                        cc_relu = LeakyReLU(cc)

                    # Combine cross correlation results with convolution of feature map A
                    netA_conv = slim.conv2d(conv_a_3, 32, 1, scope='conv_redir')
                    # Concatenate along the channels axis
                    net = tf.concat([netA_conv, cc_relu], axis=3)

                    conv3_1 = slim.conv2d(pad(net), 256, 3, scope='conv3_1')
                    with slim.arg_scope([slim.conv2d], num_outputs=512, kernel_size=3):
                        conv4 = slim.conv2d(pad(conv3_1), stride=2, scope='conv4')
                        conv4_1 = slim.conv2d(pad(conv4), scope='conv4_1')
                        conv5 = slim.conv2d(pad(conv4_1), stride=2, scope='conv5')
                        conv5_1 = slim.conv2d(pad(conv5), scope='conv5_1')
                    conv6 = slim.conv2d(pad(conv5_1), 1024, 3, stride=2, scope='conv6')
                    conv6_1 = slim.conv2d(pad(conv6), 1024, 3, scope='conv6_1')
                    ####
                    #### self attention ?
                    ####
                    ''' START: upsampling '''
                    with slim.arg_scope([slim.conv2d_transpose], biases_initializer=None):
                        deconv1_a = slim.conv2d_transpose(conv6_1, 512, 7, stride=4, padding='SAME', scope='deconv1')
                        deconv1_a = tf.layers.batch_normalization(deconv1_a, name='deconv1_a_bn', training=trainable)
                        deconv1_a = tf.nn.relu(deconv1_a, name='deconv1_a_relu')
                        deconv1_a = tf.layers.dropout(deconv1_a, rate=0.2, training=trainable)
                        deconv2_a = slim.conv2d_transpose(deconv1_a, 128, 7, 3, padding='VALID', scope='deconv2_a')
                        deconv2_a = tf.layers.batch_normalization(deconv2_a, name='deconv2_a_bn', training=trainable)
                        deconv2_a = tf.nn.relu(deconv2_a, name='deconv2_a_relu')
                        deconv2_a = tf.layers.dropout(deconv2_a, rate=0.2, training=trainable)
                        deconv3_a = slim.conv2d_transpose(deconv2_a, 32, 3, 2, padding='SAME', scope='deconv3_a')
                        deconv3_a = tf.layers.batch_normalization(deconv3_a, name='deconv3_a_bn', training=trainable)
                        deconv3_a = tf.nn.relu(deconv3_a, name='deconv3_a_relu')
                        deconv3_a = tf.layers.dropout(deconv3_a, rate=0.2, training=trainable)
                        # print deconv3_a
                        predict = slim.conv2d_transpose(deconv3_a, 1, 3, 2, padding='SAME', scope='predict')
                        predict = tf.nn.relu(predict, name='predict_relu')
                        print predict
                    return predict



    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        # inputs
        self.global_step =  tf.get_variable(
                    'global_step',
                    [],
                    initializer=tf.constant_initializer(0),
                    trainable=False
                    )
        self.left = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, self.c_dim], name='left')
        self.right = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, self.c_dim], name='right')
        self.is_training = tf.placeholder(tf.bool)
        self.ground_truth = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, 1], name='ground_truth')
        # ouput
        pred_image = self.fusenet(self.left, self.right, True)
        # pred_image = self.easynet(self.left, self.right)
        vars = tf.trainable_variables()
        # loss
        self.loss = tf.reduce_sum(tf.abs(self.ground_truth-pred_image))
        """ Training """
        self.learning_rate = tf.train.exponential_decay(self.base_lr, self.global_step, decay_steps=self.decay_steps, decay_rate=self.decay_rate, staircase=True)    

        self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate, 
                                       rho=0.95, 
                                       epsilon=1e-8,
                                       use_locking=False, 
                                       name="Adadelta").minimize(self.loss, var_list=vars)

        """ Testing """
        # pred_image = self.easynet(self.left, self.right)
        self.pred_image = self.fusenet(self.left, self.right, False)
        # self.pred_image = self.easynet(self.left, self.right, is_training=False, reuse=True)

        """ Summary """
        self.sum = tf.summary.scalar("loss", self.loss)

        tf.global_variables_initializer().run()
        print "build successfully"
        

    def predict(self, left, right, gt):
        return self.sess.run(self.pred_image, 
            feed_dict={self.left:left, self.right:right, self.ground_truth:gt, self.is_training: False})
    

    def train(self, left, right, gt):
        return self.sess.run([self.loss, self.optimizer], 
            feed_dict={self.left:left, self.right:right, self.ground_truth:gt, self.is_training: True})


