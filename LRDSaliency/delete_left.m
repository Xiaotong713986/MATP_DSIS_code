clear all
clc
video_l_path = '/data/qiudan/SVSDdataset/left_view_svsd/';
video_r_path = '/data/qiudan/SVSDdataset/right_view_svsd/';
video_l_path_dir = dir(strcat(video_l_path,'*'));
lengthVideo = length(video_l_path_dir);
index = 0;
for i = 1 : lengthVideo
    video_name = video_l_path_dir(i).name; 
%     disp(video_name)
    img_left_path = strcat(video_l_path,video_name);
    img_right_path = strcat(video_r_path,video_name);
    
    img_left_path_dir = dir(strcat(img_left_path,'/','*.jpg'));
    img_right_path_dir = dir(strcat(img_right_path,'/','*.jpg'));
    
    lengthImgL = length(img_left_path_dir);
    lengthImgR = length(img_right_path_dir);
%     index = 0;
    if(lengthImgR == lengthImgL)
        index = index + 1;
        disp('success')
    end
end
disp(index)