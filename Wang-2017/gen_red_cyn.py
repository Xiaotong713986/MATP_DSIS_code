import os
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--img", type=str, default="0")
args = parser.parse_args()


root_dir = 'OAC/Gaze_gt/'
left_dir = os.path.join(root_dir,'left')
right_dir = os.path.join(root_dir,'right')
save_dir = os.path.join(root_dir,'red-cyn')
img_name = os.listdir(left_dir)
if args.img != "0":
    img_name = [args.img]
for img in img_name:
    left_path = os.path.join(left_dir,img)
    right_path = os.path.join(right_dir,img)
    I_left = cv2.imread(left_path)
    I_right = cv2.imread(right_path)
    I_left_ = I_left[:,:,:-1]
    I_right_ = I_right[:,:,1:]
    I_merge = np.concatenate([I_left_, I_right_],2)

    #print(I_merge.shape)
    save_path = os.path.join(save_dir,img)
    cv2.imwrite(save_path,I_merge)
    print('finish the merge of {}!'.format(img))
