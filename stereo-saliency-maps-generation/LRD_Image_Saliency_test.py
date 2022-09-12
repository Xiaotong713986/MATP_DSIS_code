#import imghdr, imageio
import argparse, glob, cv2, os, numpy as np
import tensorflow as tf
import os
import time
from LRDSaliencyModel_S3D_Image import Model

fea_map_output = './output/'

#from LRDSaliencyModel import Model
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

np.set_printoptions(threshold=np.inf)
def mkDir(dirpath):
    if os.path.exists(dirpath)==0:
        os.mkdir(dirpath)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="0")
    return parser.parse_args()

args = get_arguments()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

model_path = './results_S3D/saved_model'
left_data_path = './NBU_SIRQA_origin/left/'
right_data_path = './NBU_SIRQA_origin/right/'
# save_path_all = './IDSA_res/'
save_path_all = fea_map_output
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
            print("loading checkpoint %s,waiting......", ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("load complete!")
        save_path = save_path_all + model_name_other + '_test/'
        mkDir(save_path)
        #save_fea_path = save_path_all + model_name_other + '_feature/'
        #mkDir(save_fea_path)

        left_img_name_list = os.listdir(left_data_path)
        left_img_name_list.sort()

        for index_image in range(len(left_img_name_list)):
            left_img_name = left_img_name_list[index_image]
            print(left_img_name)
            #exit()
            left_img_path = os.path.join(left_data_path,left_img_name)
            right_img_path = os.path.join(right_data_path,left_img_name)
            name_index = left_img_name.split('.')[0]
            
            left_image_path = "/data/qiudan/3DSaliency_Program/Feng_Shao_SIRQA/NBU_SIRQA_Database/all_left/12_l.png"
            right_image_path = "/data/qiudan/3DSaliency_Program/Feng_Shao_SIRQA/NBU_SIRQA_Database/all_right/12_r.png"
            #save_name = name_index+'.jpg'
            #left_files_path = glob.glob(os.path.join(left_img_path,'*.jpg'))
            save_file_path = os.path.join(save_path, name_index+'.jpg')
            save_file_path = '/data/qiudan/3DSaliency_Program/Test_20201028/000010_LRD_sal.jpg'

            left = left_image_path
            left_name=os.path.basename(left)
    
            frame_left = cv2.imread(left)
            frame_left = frame_left[:, :, ::-1]
            frame_left = frame_left- MEAN_VALUE
            frame_left = cv2.resize(frame_left, (112,112))
            frame_left = frame_left / 255.
            right = right_image_path

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
            #[sal_map,concat,conv_out1,conv_out2,conv_out3,conv_out4,conv_out5] =  
            predictions = sess.run(sal, feed_dict={x: frame_left,y: frame_right,is_training:False})
            sal_map = predictions[0]
            for ii in range(11):
                fea_map = predictions[ii][0,:,:,0]

                fea_map = cv2.resize(fea_map,(112,112))
                fea_map = (fea_map-np.amin(fea_map)) /( np.amax(fea_map)-np.amin(fea_map))
                fea_map = fea_map*255
                fea_map = cv2.applyColorMap( np.uint8(fea_map),cv2.COLORMAP_JET)

                cv2.imwrite(fea_map_output + '%s_conv_out_%s.jpg' % (name_index,ii), fea_map)
            print('############################')
            elapsed = (time.clock() - start)
            print("Time used:",elapsed)

            print('sal_map shape:',sal_map.shape)
            save_image = np.zeros([112, 112])
            test = sal_map[0, :, :, 0]#*255.
            test = test -np.amin(test)
            test = test /np.amax(test)
            test *=255.
            
            print(save_file_path)
            cv2.imwrite(save_file_path, test)

            print("The %d image Done.",index_image)

            
                    
                

