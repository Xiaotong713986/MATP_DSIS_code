#!/usr/bin/env python3
# encoding: utf-8
import cv2
import tensorflow as tf
import sys, os
import numpy as np
import tensorflow.contrib.layers as layers
# qiudan
import cPickle as pkl
from DatasetLR import VideoDataset
from LRDSaliencyAtten_2 import Model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse, cv2, os, glob, sys, time
import cPickle as pkl
from utils.pymetric.metrics import CC, SIM, AUC_Judd


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='training gpu')
    parser.add_argument('--plotiter', type=int, default=500, help='training mini batch')
    parser.add_argument('--validiter', type=int, default=4000, help='training mini batch')
    parser.add_argument('--savemodeliter', type=int, default=1500, help='training mini batch')
    parser.add_argument('--snapshotincode', type=bool, default=False, help='save snapshot in code')
    
    parser.add_argument('--trainingexampleprops',type=float, default=0.9, help='training dataset.')
    #parser.add_argument('--trainingbase',type=str, default='svsd', help='training dataset.')
    parser.add_argument('--videolength',type=int,default=16, help='length of video')
    parser.add_argument('--overlap',type=int,default=15, help='dataset overlap')
    parser.add_argument('--batch',type=int,default=2, help='length of video')
    parser.add_argument('--imagesize', type=tuple, default=(112,112))
    
    parser.add_argument('--extramodinfo', type=str, default='', help="add extra model information")
    return parser.parse_args()

 
print "loading data and lable ... "
train_frame_basedir_left = '/data/qiudan/SVSDdataset/left_view_svsd'
train_frame_basedir_right = '/data/qiudan/SVSDdataset/right_view_svsd'

train_density_basedir = '/data/qiudan/SVSDdataset/left_density_svsd'
all_path = './results_svs'
SaveFile_name = 'save_svs_models_atten_8w'
SaveFile = os.path.join(all_path, SaveFile_name)
if not os.path.isdir(SaveFile):
    os.makedirs(SaveFile)
def main():
    print ("Parsing arguments...")
    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    '''A2B: Add postfix to identify a model version'''
    postfix_str="figure_svs_atten_8w"
    batch_size = args.batch
    #training_base = args.trainingbase
    video_length=args.videolength
    image_size = args.imagesize 
    ##
    ##  data
    print ("Loading data...")
    tranining_dataset = VideoDataset(train_frame_basedir_left,train_frame_basedir_right, train_density_basedir, img_size=(112,112), bgr_mean_list=[98,102,90], sort='rgb')
    tranining_dataset.setup_video_dataset_c3d(overlap=args.overlap, training_example_props=args.trainingexampleprops)
    ##
    ##  保存figures
    plot_figure_dir = all_path
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


    max_iter = 80000
    validation_iter = args.validiter
    plot_iter = args.plotiter
    epoch=30
    idx_counter = 0
    save_model_iter = args.savemodeliter
    momentum = 0.95
    weight_decay = 0.0005
    init_learning_rate = 1e-2
    
    is_training = tf.placeholder(tf.bool)
    inputs_left = tf.placeholder(tf.float32, [batch_size, 112, 112, 3])
    inputs_right = tf.placeholder(tf.float32, [batch_size, 112, 112, 3])
    ground_truth = tf.placeholder(tf.float32, [batch_size, 112, 112, 1])

    # start training...
    pred = Model.LRDSaliency_inference(inputs_left,inputs_right,is_training)
    #print 'pred:', pred.shape
    #pred_reshape = tf.reshape(pred, [batch_size, video_length, 112, 112])
    #print 'pred_reshape:', pred_reshape.shape
    #exit()
    with tf.name_scope('loss'):    
        # L1 loss in c3dsaliency
        loss = tf.reduce_mean(tf.reduce_sum(tf.abs(pred - ground_truth)))
        tf.summary.scalar('loss', loss)
    
    with tf.name_scope('train'):    
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=init_learning_rate, rho=momentum, epsilon=weight_decay).minimize(loss)           
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
        with tf.control_dependencies(update_ops):
            train_op=tf.group(optimizer)
    #saver = tf.train.Saver()   

    with tf.Session() as sess:

        summary_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter("logs/", sess.graph)

        init = tf.global_variables_initializer()
        print "Initialize the variables"
        sess.run(init)
        
        #Until Now, the pretrained model is not found
        #saver.restore(sess,"./pretrain_model.ckpt") 
        
        saver = tf.train.Saver(tf.global_variables())
        print "The training stage begins"
        
        _step = 0
        
        while _step < max_iter:
            frame_data_left, frame_data_right, density_data = tranining_dataset.get_frame_LR(mini_batch=batch_size, phase='training', density_length='full')
            summary, L1_loss, _ = sess.run([summary_op, loss, train_op], feed_dict={inputs_left: frame_data_left, inputs_right: frame_data_right, ground_truth: density_data,is_training: True})
            #print (type(density_data), type(frame_data))
            print 'The training loss:', L1_loss
            #print 'The Learning Rate of each step is ', sess.run(optimizer.init_learning_rate)
            if _step % 2000 ==0:
                saver.save(sess, SaveFile + '/' + 'train_IDSENet_SVS_',global_step = _step)
                summary, result_loss, _ = sess.run([summary_op, loss, train_op],feed_dict={inputs_left: frame_data_left, inputs_right: frame_data_right, ground_truth: density_data,is_training: True})
                assert not np.isnan(result_loss), 'Model diverged with loss = NaN'
                writer.add_summary(summary, _step)

            plot_dict['x'].append(_step)
            plot_dict['y_loss'].append(L1_loss)

            if _step % validation_iter==0:
                print "Doing validation...", tranining_dataset.num_validation_examples, "validation samples in total."
                tmp_cc = []; tmp_sim = []; tmp_auc = []
                data_tuple = tranining_dataset.get_frame_LR(mini_batch=batch_size, phase='validation', density_length='one')
                index = 0
                while data_tuple is not None:
                    print index,'\r',
                    sys.stdout.flush()
                    index += 1
                    valid_frame_data_left, valid_frame_data_right, valid_density_data = data_tuple
                    pred_valid = sess.run(pred, feed_dict ={inputs_left: valid_frame_data_left, inputs_right: valid_frame_data_right,is_training: False})
                    # updated.
                    pred_valid = np.reshape(pred_valid, [-1, 1, 112, 112])
                    valid_density_data = np.reshape(valid_density_data, [-1, 1, 112, 112])
                    # pre_valid [2, 16, 112, 112] valid_density_data [2, 1, 112, 112]
                    for (prediction, valid_density) in zip(pred_valid, valid_density_data):
                        # print np.array(prediction).shape, np.array(valid_density).shape
                        preds = np.array(prediction[-1])
                        gt = np.array(valid_density[-1])                        
                        tmp_cc.append(CC(preds, gt))
                        tmp_sim.append(SIM(preds, gt))
                        tmp_auc.append(AUC_Judd(preds, gt))
                        #print('The validation CC:',CC(preds, gt))
                        #print('The validation SIM:',SIM(preds, gt))
                    data_tuple = tranining_dataset.get_frame_LR(mini_batch=batch_size, phase='validation', density_length='one')
            
                tmp_cc = np.array(tmp_cc)[~np.isnan(tmp_cc)]
                tmp_sim = np.array(tmp_sim)[~np.isnan(tmp_sim)]
                tmp_auc = np.array(tmp_auc)[~np.isnan(tmp_auc)]
                print "CC, SIM, AUC_JUDD: ", np.mean(tmp_cc), np.mean(tmp_sim), np.mean(tmp_auc)
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
            
            _step += 1


if __name__ == "__main__":
    main()