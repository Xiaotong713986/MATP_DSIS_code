from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os



def my_resize(mask_w, mask_l, img):
    ##### img is Image type image #########
    width = img.size[0]
    length = img.size[1]

    ############### downsample firstly ###############
    im_down = img.resize((int(width /2), length), Image.ANTIALIAS)
    width = int(width / 2)
    ################################################

    if width > mask_w:
        new_length = int(length * (mask_w / width))
        im_resize = im_down.resize((mask_w, new_length), Image.ANTIALIAS)
        length = new_length
        width = mask_w
    elif length > mask_l:
        new_width = int(width * (mask_l / length))
        im_resize = im_down.resize((new_width, mask_l), Image.ANTIALIAS)
        width = new_width
        length = mask_l
    else:
        im_resize = im_down
    im_down = im_resize
    return (im_down, width, length)

def combine_photo(scr_width, scr_length, arr, flag):
    # combine image from two image
    #flag = 0
    # combine image from one image, remerge
    #flag = 1
    if flag ==0:
        img1 = Image.open(arr[0])
        img2 = Image.open(arr[1])
    else:
        img = Image.open(arr)
        width = img.size[0]
        length = img.size[1]
        img1 = img.crop([0,0,int(width/2),length])
        img2 = img.crop([int(width/2),0,width,length])

    img1_ = my_resize(int(scr_width/2), scr_length, img1)
    img2_ = my_resize(int(scr_width/2), scr_length, img2)
    '''
    print((img1_[1], img1_[2]))
    print((img2_[1], img2_[2]))
    print(img1_[0].width)
    '''
    toImage = Image.new('RGB', (scr_width, scr_length))
    toImage.paste(img1_[0], (int(scr_width/4-(img1_[1])/2), int(scr_length/2-(img1_[2])/2)))
    toImage.paste(img2_[0], (int(3*scr_width / 4 - (img2_[1])/2), int(scr_length / 2 - (img2_[2])/2)))
    return toImage

def combine_photo2(scr_width, scr_length, arr, flag):
    # combine image from two image
    #flag = 0
    # combine image from one image, remerge
    #flag = 1
    if flag ==0:
        img1 = Image.open(arr[0])
        img2 = Image.open(arr[1])
    else:
        img = Image.open(arr)
        width = img.size[0]
        length = img.size[1]
        img1 = img.crop([0,0,int(width/2),length])
        img2 = img.crop([int(width/2),0,width,length])

    img1_ = img1.size
    img2_ = img2.size 
    '''
    print((img1_[1], img1_[2]))
    print((img2_[1], img2_[2]))
    print(img1_[0].width)
    '''
    toImage = Image.new('RGB', (scr_width, scr_length))
    toImage.paste(img1, (int(scr_width/4-(img1_[0])/2), int(scr_length/2-(img1_[1])/2)))
    toImage.paste(img2, (int(3*scr_width / 4 - (img2_[0])/2), int(scr_length / 2 - (img2_[1])/2)))
    return toImage

    
if __name__ == '__main__':
    #root_dir = 'C:/Users/lay/Desktop/eye_tracker_database/QIUDANG-image'
    root_dir= '/data/qiudan/3DSaliency_Program/Cropping/Wang-2017-Stcvessd--master/OAC/Wang_sal/'
    left_dir = os.path.join(root_dir,'left')
    right_dir = os.path.join(root_dir,'right')
    save_dir = "/data/qiudan/3DSaliency_Program/Cropping/Wang-2017-Stcvessd--master/OAC/Wang_sal/OAC-Wang-display/"
    # combine image from two image
    flag = 0
    # combine image from one image, remerge
    
    #path = os.path.join(root_dir, fold)
    #img_list = os.listdir(path)
    #print(img_list)

    #output_dir = '/data/xiaoxiaotong/lxl_database/previous-output'
    img_list = os.listdir(left_dir)

    if flag == 0:
        for i in range(len(img_list)):
            img1_path = os.path.join(left_dir, img_list[i])
            img2_path = os.path.join(right_dir, img_list[i])
            img_path = [img1_path, img2_path]
            comb_img = combine_photo2(1920, 1080, img_path, 0)
            img_name = img_list[i]
            img_output = os.path.join(save_dir, img_name)
            comb_img.save(img_output, quality=95)
            print('combine '+ img_name+' finish!')
    else:
        for i in range(0, len(img_list)):
            img_path = os.path.join(path, img_list[i])
            comb_img = combine_photo(1920, 1080, img_path, 1)
            img_name = img_list[i].split('.')[0]+'_combi.png'
            img_output = os.path.join(output_dir, img_name)
            comb_img.save(img_output, quality=95)
            print('combine '+ img_name+' finish!')
     
