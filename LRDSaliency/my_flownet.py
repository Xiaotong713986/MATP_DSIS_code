#!/usr/bin/env python3
# encoding: utf-8\
import cv2
import tensorflow as tf
import sys, os, h5py
import numpy as np
import tensorflow.contrib.layers as layers
import random
import  pandas as pd
from random import shuffle
from random import randint
from tqdm import  tqdm
import time

# qiudan
import cPickle as pkl
from DatasetLR import VideoDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse, cv2, os, glob, sys, time, datetime
import cPickle as pkl
import google.protobuf.text_format as txtf
# from validation import MetricValidation
from utils.pymetric.metrics import CC, SIM, AUC_Judd
from PIL import Image
from Flownet import *


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

HEADLESS = False
if HEADLESS:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Use the custom correlation layer or build one from tensorflow slice operations
use_custom_correlation = False
if use_custom_correlation:
  import correlation_layer as cl

QUICK_DEBUG = True
num_epochs = 10
num_train_sets = 9
loss_order = 2


class TTT:
    def __init__(self,
            keep_prob = 0.6,
            batch_size = 8,
            epoch=40):
        # source image
        self.BATCH_SIZE = batch_size
        self.MAX_ITER = 300000
        self.CROP_WIDTH = 112
        self.CROP_HEIGHT = 112
        self.graph = tf.Graph()
        self.epoch = epoch
        # frame number = 8
        self.CLIP_LENGTH = 1
        decay_epoch=10   #每5个epoch改变一次学习率
        self.n_step_epoch = 50

    def train(self):
        print ("Parsing arguments...")
        args = get_arguments()
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        snapshot_path = args.use_snapshot
        #Check if snapshot exists
        if snapshot_path is not '':
            if not os.path.isfile(snapshot_path):
                print (snapshot_path, "not exists.Abort")
                exit()
        # Save models Settings  
        premodel_save_dir = './pretrain/'
        '''
        postfix and prefix
        '''
        '''A2B: Add postfix to identify a model version'''
        postfix_str="fusenet_0.1_pretrained"
        #exit()
        batch = args.batch
        training_base = args.trainingbase
        video_length=args.videolength
        image_size = args.imagesize 
        ##
        ##  data
        ##
        print ("Loading data...")
        if training_base=='svsd':
            train_frame_basedir = '/data/qiudan/videosaliency/videodataset/SVSDataset/left-view'
            train_frame_basedir_right = '/data/qiudan/videosaliency/videodataset/SVSDataset/right-view'
            train_density_basedir = '/data/qiudan/videosaliency/videodataset/SVSDataset/svsd_density/'

        tranining_dataset = VideoDataset(train_frame_basedir,train_frame_basedir_right, train_density_basedir, img_size=(112,112), bgr_mean_list=[98,102,90], sort='rgb')
        tranining_dataset.setup_video_dataset_c3d(training_example_props=args.trainingexampleprops)

        plot_figure_dir = './figure'
        ## Figure dir
        plot_figure_dir = os.path.join(plot_figure_dir, postfix_str)
        if not os.path.isdir(plot_figure_dir):
            os.makedirs(plot_figure_dir)
        print ("Loss figure will be save to", plot_figure_dir)


        plot_dict = {
        'x':[], 
        'x_valid':[], 

        'y_loss':[], 
        'y_cc':[], 
        'y_sim':[], 
        'y_auc':[]
        }

        plt.subplot(4, 1, 1)
        plt.plot(plot_dict['x'], plot_dict['y_loss'])
        plt.ylabel('loss')
        plt.subplot(4, 1, 2)
        plt.plot(plot_dict['x_valid'], plot_dict['y_cc'])
        plt.ylabel('cc metric')
        plt.subplot(4, 1, 3)
        plt.plot(plot_dict['x_valid'], plot_dict['y_sim'])
        plt.ylabel('sim metric')
        plt.subplot(4, 1, 4)
        plt.plot(plot_dict['x_valid'], plot_dict['y_auc'])
        plt.xlabel('iter')
        plt.ylabel('auc metric')


        max_iter = 500000
        validation_iter = args.validiter
        plot_iter = args.plotiter
        save_model_iter = args.savemodeliter
        momentum = 0.9
        weight_decay = 0.0005
        base_lr = 1e-3
        # start training...
        
        t = datetime.datetime.now().isoformat()[:-9]
        dir_name = training_base + '_' + str(max_iter) + '_' + str(base_lr)  +'_'+ t +'/'

        plot_figure_dir = './figure/' + dir_name
        if not os.path.exists(plot_figure_dir):
            os.mkdir(plot_figure_dir)
            print ("plot_figure_dir created. Loss figure will be save to", plot_figure_dir)
        model_save_dir = './models/' + dir_name
        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)
            print ("model_save_dir created. Models will be save to", model_save_dir)

        with self.graph.as_default():
            with tf.Session(config=config, graph=self.graph) as sess:
                _step = 0
                flownet = Flownet(sess, learning_rate=1e-2, batch_size=8)
                flownet.build_model()
                # exit()
                saver = tf.train.Saver()
                # loading pre_train_models only fusenet            
                variable_map = {}
                for variable in tf.global_variables():
                    if 'global' not in variable.name \
                        and 'upsample' not in variable.name.split('/')[1] \
                        and 'predict' not in variable.name.split('/')[1] \
                        and 'deconv' not in variable.name.split('/')[1] :
                    #     # and 'Mixed_4' not in variable.name.split('/')[2]:
                    #     # delete high-level features
                        # print variable.name
                        variable_map[variable.name.replace(':0', '')] = variable
                saver = tf.train.Saver(var_list=variable_map, reshape=True)
                ckpt = tf.train.get_checkpoint_state(premodel_save_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    print("loading checkpoint %s,waiting......" % ckpt.model_checkpoint_path)
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print("load complete!")
                all_saver = tf.train.Saver(tf.global_variables())
                # Start training...
                while _step < max_iter:
                    start_time = time.time()
                    frame_data, frame_data_right, density_data  = tranining_dataset.get_frame_LR_tf(mini_batch=self.BATCH_SIZE, phase='training', density_length='full')
                    # update network
                    loss, _ = flownet.train(frame_data, frame_data_right, density_data)


                    duration = time.time() - start_time
                    if (_step < 200 and _step % 10 ==0) or  _step % 1000 == 1 :
                        print (datetime.datetime.now().isoformat()[:-7],' Step %d: %.3f sec, Loss: %.3f' % (_step, duration, np.sum(loss)))
                    plot_dict['x'].append(_step)
                    plot_dict['y_loss'].append(loss)

                    if _step % validation_iter==0:
                        print "Doing validation...", tranining_dataset.num_validation_examples, "validation samples in total."
                        tmp_cc = []; tmp_sim = []; tmp_auc = []
                        data_tuple = tranining_dataset.get_frame_LR_tf(mini_batch=self.BATCH_SIZE, phase='validation', density_length='one')
                        index = 0
                        while data_tuple is not None:
                            # print index,'\r',
                            # sys.stdout.flush()
                            index += 1
                            valid_frame_data,valid_frame_data_right, valid_density_data = data_tuple
                            # print 'valid_frane', np.sum(sess.run(self.ImageL)), 'valid_density', np.sum(sess.run(self.ImageR))
                            predictions_valid = flownet.predict(valid_frame_data,valid_frame_data_right, valid_density_data)
                            # print np.array(predictions_valid).shape                            
                            for (prediction, ground_truth) in zip(predictions_valid, valid_density_data):
                                ## shape like this "h, w ,c"
                                prediction = np.transpose(np.array(prediction), (2, 0, 1));
                                ground_truth = np.transpose(np.array(ground_truth), (2, 0, 1));
                                
                                
                                if index % 500 == 0:
                                    print 'index', index,'valid_left', np.sum(valid_density_data), 'valid_pred', np.sum(np.abs(predictions_valid)), 'loss', np.sum(np.abs(valid_density_data-predictions_valid))
                                    save_image1 = np.zeros([112, 112])
                                    save_image1[:, :] = prediction[0, :, :]
                                    final_save_image1 = save_image1*255.
                                    # print final_save_image1
                                    # if step%10==0:
                                    smap_save_path = os.path.join('./', 'smap_Result')
                                    smap_save_name = smap_save_path + '/' +  'step' + str(index) + 'frame' + str(0) + '.jpg'
                                    cv2.imwrite(smap_save_name, final_save_image1)
                                # exit()
                                if index % 500 == 1:
                                    print datetime.datetime.now(), ' Index', index, 'pred:', np.sum(prediction), 'gt:', np.sum(ground_truth), 'loss:', np.sum(np.abs(prediction-ground_truth))
                                for (pred, gt) in zip(prediction, ground_truth):
                                    tmp_cc.append(CC(pred, gt))
                                    tmp_sim.append(SIM(pred, gt))
                                    tmp_auc.append(AUC_Judd(pred, gt))  
                            data_tuple = tranining_dataset.get_frame_LR_tf(mini_batch=self.BATCH_SIZE, phase='validation', density_length='one')
                        tmp_cc = np.array(tmp_cc)[~np.isnan(tmp_cc)]
                        tmp_sim = np.array(tmp_sim)[~np.isnan(tmp_sim)]
                        tmp_auc = np.array(tmp_auc)[~np.isnan(tmp_auc)]
                        print datetime.datetime.now().isoformat()[:-7], " Step:", _step, " Metrics:", np.mean(tmp_cc), np.mean(tmp_sim), np.mean(tmp_auc)
                        plot_dict['x_valid'].append(_step)
                        plot_dict['y_cc'].append(np.mean(tmp_cc))
                        plot_dict['y_sim'].append(np.mean(tmp_sim))
                        plot_dict['y_auc'].append(np.mean(tmp_auc))
                        
                    if _step%plot_iter==0:
                        plot_xlength=500
                        plt.subplot(4, 1, 1)
                        plt.plot(plot_dict['x'][-plot_xlength:], plot_dict['y_loss'][-plot_xlength:])
                        plt.ylabel('loss')
                        plt.subplot(4, 1, 2)
                        plt.plot(plot_dict['x_valid'][-plot_xlength:], plot_dict['y_cc'][-plot_xlength:])
                        plt.ylabel('cc metric')
                        plt.subplot(4, 1, 3)
                        plt.plot(plot_dict['x_valid'][-plot_xlength:], plot_dict['y_sim'][-plot_xlength:])
                        plt.ylabel('sim metric')
                        plt.subplot(4, 1, 4)
                        plt.plot(plot_dict['x_valid'][-plot_xlength:], plot_dict['y_auc'][-plot_xlength:])
                        plt.xlabel('iter')
                        plt.ylabel('auc metric')
                        plt.savefig(os.path.join(plot_figure_dir, "plot"+str(_step)+".png"))
                        plt.clf()


                        pkl.dump(plot_dict, open(os.path.join(plot_figure_dir, "plot_dict.pkl"), 'wb'))
                    
                    # 保存变量
                    all_saver.save(sess, model_save_dir, global_step=5000, write_meta_graph=False)

                    _step += 1


def get_arguments():
    parser = argparse.ArgumentParser()
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

if __name__ == '__main__':
    model = TTT()
    model.train()