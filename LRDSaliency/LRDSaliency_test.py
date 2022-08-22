#import imghdr, imageio
from math import floor
import argparse, glob, cv2, os, numpy as np, sys
import tensorflow as tf
import sys
import os
import time
#from LRDSaliencyModel import Model
from LRDSaliencyModel import Model
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def mkDir(dirpath):
    if os.path.exists(dirpath)==0:
        os.mkdir(dirpath)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="0")
    return parser.parse_args()

args = get_arguments()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

model_path ='/data/qiudan/LRDSaliency/results_svs/save_model_627/'
left_data_path = '/data/qiudan/SVSDdataset/SVSDTest/left_view_svsd-test'
right_data_path = '/data/qiudan/SVSDdataset/SVSDTest/right_view_svsd'
save_path_all = '/data/qiudan/LRDSaliency/results_svs/save_model_627/'
if os.path.isdir(save_path_all):
    pass
else:
    os.mkdir(save_path_all)

model_file = glob.glob(model_path+"*.data-00000-of-00001")

MEAN_VALUE = np.array([103.939, 116.779, 123.68], dtype=np.float32)
is_training = tf.placeholder(tf.bool)
x = tf.placeholder(tf.float32, [1, 112, 112, 3])
y = tf.placeholder(tf.float32, [1, 112, 112, 3])
sal = Model.LRDSaliency_inference(x,y,is_training)

with tf.Session(config=config) as sess:
    for j in range(len(model_file)):
        #print j

        model = model_file[j]
        model_name = os.path.basename(model)
        #print('model_name:',model_name)
        #exit()
        model_name_other = model_name.split('.')[0]
        model_name_path = model_name_other+'.ckpt'
        # restore model
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess.run(init)
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            print("loading checkpoint %s,waiting......" % ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("load complete!")
        save_path = save_path_all + model_name_other + '_test/'
        mkDir(save_path)
        #save_fea_path = save_path_all + model_name_other + '_feature/'
        #mkDir(save_fea_path)

        left_folds_name_list = os.listdir(left_data_path)
        left_folds_name_list.sort()

        for index_video in range(len(left_folds_name_list)):
            left_fold_name = left_folds_name_list[index_video]
            print(left_fold_name)
            
            left_fold_path = os.path.join(left_data_path,left_fold_name)
            right_fold_path = os.path.join(right_data_path,left_fold_name)

            left_files_path = glob.glob(os.path.join(left_fold_path,'*.jpg'))
            save_file_path = save_path + left_fold_name
            #save_feature_path = save_fea_path + left_fold_name
            mkDir(save_file_path)
            #mkDir(save_feature_path)

            for i in range(len(left_files_path)):
                # for each video
                left = left_files_path[i]
                left_name = os.path.basename(left)#.split('.')[0]
                #name_index = int(left_name.split('_')[1])
                #frame_wildcard = "frame_%d.jpg"
                #start_frame_index = name_index 
                #end_frame_index = name_index + index
                #current_frame_list = []
        
                frame_left = cv2.imread(left)
                frame_left = frame_left[:, :, ::-1]
                frame_left = frame_left- MEAN_VALUE
                frame_left = cv2.resize(frame_left, (112,112))
                frame_left = frame_left / 255.
                print('frame_left:',frame_left)
                right = os.path.join(right_fold_path,left_name)
                print('left:',left)
                print('right:',right)
                #exit()
                frame_right = cv2.imread(right)
                frame_right = frame_right[:, :, ::-1]
                frame_right = frame_right- MEAN_VALUE
                frame_right = cv2.resize(frame_right, (112,112))
                frame_right = frame_right / 255.      
                #print current_frame_list.shape
                frame_left = np.array(frame_left)[None, :]
                frame_right = np.array(frame_right)[None, :]
                print('frame_left:',frame_left.shape)
                start = time.clock()                
                sal_map = sess.run(sal, feed_dict={x: frame_left,y: frame_right,is_training:False})
                elapsed = (time.clock() - start)
                print("Time used:",elapsed)
                #print sal_map.shape
                #print sal_map
                exit()
                #sal_map_single = sal_map[0,m,:,:,0]
                #print sal_map_single.shape
                #print sal_map_single
                #exit()
                #image = np.reshape(sal_map, [-1, 1, 112, 112]) 
                save_image = np.zeros([112, 112])
                test = sal_map[0, :, :, 0]#*255.
                test = test -np.amin(test)
                test = test /np.amax(test)
                test *=255.
                #print save_image.shape
                #print save_image
                #exit()
                save_name = os.path.join(save_file_path, left_name)
                cv2.imwrite(save_name, test)

            print "The %d video Done." % index_video
                    
                

