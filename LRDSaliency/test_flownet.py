# -*- coding:gb2312 -*-
import tensorflow as tf
import sys
import os

from Deep3DSaliency_model import Model
from network import *
from datetime import datetime
import cv2
import math
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import time
from DatasetLR import VideoDataset
import numpy as np

from PIL import Image
from Flownet import *
from metrics import CC, SIM, AUC_Judd, KLdiv, NSS, AUC_Borji

use_custom_correlation = False
if use_custom_correlation:
  import correlation_layer as cl


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='flownet', help="model")
    parser.add_argument('--use_snapshot', type=str, default='', help='Snapshot path.')
    parser.add_argument('--plotiter', type=int, default=500, help='training mini batch')
    parser.add_argument('--validiter', type=int, default=4000, help='training mini batch')
    parser.add_argument('--savemodeliter', type=int, default=1500, help='training mini batch')

    parser.add_argument('--trainingexampleprops',type=float, default=0.8, help='training dataset.')
    parser.add_argument('--trainingbase',type=str, default='svsd', help='training dataset.')
    parser.add_argument('--videolength',type=int,default=16, help='length of video')
    parser.add_argument('--overlap',type=int,default=1, help='dataset overlap')
    parser.add_argument('--batch',type=int,default=2, help='length of video')
    parser.add_argument('--imagesize', type=tuple, default=(112,112))
    parser.add_argument('--gpu', type=str, default="0")
    
    parser.add_argument('--extramodinfo', type=str, default='', help="add extra model information")
    return parser.parse_args()

def generate_saliency(pred_patch_smap):
    # pred_patch_l shape: [patch_no, 224, 224, 1]
    boxsize=224
    patches=pred_patch_smap.shape[0]

    pa1 = int(math.ceil(1080 / boxsize)+1)
    pa2 = int(math.ceil(1920 / boxsize)+1)
    final_smap=np.zeros([1080,1920])
    temp_smap=np.zeros([pa1*boxsize,pa2*boxsize])

    patch_no=0
    for i in range(pa1):
        for j in range(pa2):
            temp_patch=pred_patch_smap[patch_no, :, :]
            temp_smap[boxsize*i:boxsize*(i+1), boxsize*j:boxsize*(j+1)]=temp_patch
            final_smap[:,:]=temp_smap[0:1080, 0:1920]
            patch_no=patch_no+1
    return final_smap

def mkDir(dirpath):
    if os.path.exists(dirpath)==0:
        os.mkdir(dirpath)

args = get_arguments()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print ("Loading data...")
LeftPath = '../svsd/SVS-Test/left_view_svsd/'
RightPath = '../svsd/SVS-Test/right_view_svsd/'
GtPath = '../svsd/SVS-Test/view_svsd_density/'


# LeftPath = '../svsd/left-view'
# RightPath = '../svsd/right-view'
# GtPath = '../svsd/svsd_density/'



tranining_dataset = VideoDataset(LeftPath, RightPath, GtPath, img_size=(112, 112),   bgr_mean_list=[98,102,90], sort='rgb')
tranining_dataset.setup_video_dataset_c3d(training_example_props=0)

print "data loaded"
trainflag = args.net

batch_size=1
frames=1

model_name = trainflag + '/'


with tf.Session(config=config) as sess:
    if int((tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
        merged = tf.merge_all_summaries()
    else:  # tensorflow version >= 0.12
        merged = tf.summary.merge_all()
    
    logs_dir = './logs/' + dir_name
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    # tf.train.SummaryWriter soon be deprecated, use following
    if int((tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
        writer = tf.train.SummaryWriter(logs_dir, sess.graph)
    else:  # tensorflow version >= 0.123
        writer = tf.summary.FileWriter(logs_dir, sess.graph)

    if int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()

    print 'Init variable'

    saver = tf.train.Saver()

    sess.run(init)
    
    flownet = Flownet(sess, learning_rate=1e-2)
    flownet.build_model()
    variable_map = {}
    for variable in tf.global_variables():
        variable_map[variable.name.replace(':0', '')] = variable
    saver = tf.train.Saver(var_list=variable_map, reshape=True)

    ckpt = tf.train.get_checkpoint_state("./model/"+model_name)
    if ckpt and ckpt.model_checkpoint_path:
        print("loading checkpoint %s,waiting......" % ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("load complete!")
            
    print "Doing test...", tranining_dataset.num_validation_examples, " samples in total."
    # KLdiv, NSS, AUC_Borji
    tmp_cc = []; tmp_sim = []; tmp_auc = []; tmp_kld = []; tmp_nss = []; tmp_aucborji = [];
    data_tuple = tranining_dataset.get_frame_LR_tf(mini_batch=batch_size, phase='validation', density_length='one')
    index = 0
    while data_tuple is not None:
        test_l, test_r, test_gt = data_tuple
        index += 1
        image = flownet.predict(test_l, test_r, test_gt)
        print datetime.datetime.now(), ' Index', index, 'pred:', np.sum(iamge), \
            'gt:', np.sum(test_gt)
        # validation
        image1 = np.reshape(image, [-1, 1, 224, 224]) 

        # for i in range(frames):
        save_image1 = np.zeros([224, 224])
        save_image2 = np.zeros([224, 224])

        save_image1[:, :] = image1[0, 0, :, :]*255.
        save_image2 = valid_batch_ys[0][0]*255.
        final_save_image1 = save_image1
        
        print np.sum(save_image1),  np.sum(save_image2)
        # print final_save_image1
        final_save_image2 = save_image2
        # if step%10==0:

        smap_save_path = './test'
        mkDir(smap_save_path)
        smap_save_name = smap_save_path + '/' +  'index_' + str(index) + '_pred.jpg'
        cv2.imwrite(smap_save_name, final_save_image1)
        smap_save_name2 = smap_save_path + '/' +  'index_' + str(index) + '_gt.jpg'
        cv2.imwrite(smap_save_name2, final_save_image2)
        for (prediction, ground_truth) in zip(image0, test_gt):
            prediction = np.transpose(np.array(prediction[0]), (2, 0, 1))
            ground_truth = np.array(ground_truth)
            for (preds, gt) in zip(prediction, ground_truth):
                tmp_cc.append(CC(preds, gt))
                tmp_sim.append(SIM(preds, gt))
                tmp_auc.append(AUC_Judd(preds, gt))  
                tmp_nss.append(NSS(preds, gt))
                tmp_kld.append(KLdiv(preds, gt))
                tmp_aucborji.append(AUC_Borji(preds, gt))
        data_tuple = tranining_dataset.get_frame_LR_tf(mini_batch=batch_size, phase='validation', density_length='one')
    tmp_cc = np.array(tmp_cc)[~np.isnan(tmp_cc)]
    tmp_sim = np.array(tmp_sim)[~np.isnan(tmp_sim)]
    tmp_auc = np.array(tmp_auc)[~np.isnan(tmp_auc)]
    tmp_nss = np.array(tmp_nss)[~np.isnan(tmp_nss)]
    tmp_kld = np.array(tmp_kld)[~np.isnan(tmp_kld)]
    tmp_aucborji = np.array(tmp_aucborji)[~np.isnan(tmp_aucborji)]
    print " Step: %d, Metrics: CC: %.3f  SIM: %.3f  AUC_Judd: %.3f  NSS: %.3f  KLdiv: %3f  AUC_Borji: %.3f" \
        % (index, np.mean(tmp_cc), np.mean(tmp_sim), np.mean(tmp_auc), np.mean(tmp_nss), np.mean(tmp_kld), np.mean(tmp_aucborji))

    print 'Testing Finished!'
