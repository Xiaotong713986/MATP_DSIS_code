% ==== (SNS matlab code)======
% The Code (Version 1) is created by ZHANG Yabin,
% Nanyang Technological University, 2015-12-30
% which is based on the method described in the following paper 
% [1] Wang, Yu-Shuen, et al. "Optimized scale-and-stretch for image resizing." 
% ACM Transactions on Graphics (TOG) 27.5 (2008): 118. 
% The binary code is provided on the project page:
% http://graphics.csie.ncku.edu.tw/Image_Resizing/
% The Matlab codes are for non-comercial use only.
% Note that the importance maps are slightly different from the original
% ones, and the retargeted images are influenced.

clear all; clc
addpath(genpath('SIFT_'));
imL = imread('C:\Users\lay\Desktop\ICME2020-xxt\»­Í¼\Enchanted_pic-0-54-42_left.jpg');
imR = imread('C:\Users\lay\Desktop\ICME2020-xxt\»­Í¼\Enchanted_pic-0-54-42_right.jpg');
%im_SNS = imread('tajmahal_0.50_sns.png');

% im =  imread('Brasserie_L_Aficion.png');
% im_SNS = imread('Brasserie_L_Aficion_0.50_sns.png');

% parameters
ALIGIN_TERM = 1;
Ratio = 0.5;
hRatio = 0.5;
mesh_size = 20; % using mesh_size x mesh_size quad
[h, w, ~] = size(imL);
quad_num_h = floor(h/mesh_size);
quad_num_w = floor(w/mesh_size);




% the importance map generation
% importance_map =  SNS_importanceMap(imL, true); % generate the importance map
% importance_mapR =  SNS_importanceMap(imR, true);
importance_map = double(imread('C:\Users\lay\Desktop\ICME2020-xxt\»­Í¼\Enchanted_pic-0-54-42_sal_left.bmp'));
importance_mapR = double(imread('C:\Users\lay\Desktop\ICME2020-xxt\»­Í¼\Enchanted_pic-0-54-42_sal_right.bmp'));

%%% crop first
[new_imgL, new_imgR, importance_map, importance_mapR] =cutImg2(imL, imR, importance_map, importance_mapR, Ratio, 1);
imwrite(new_imgL,'C:\Users\lay\Desktop\ICME2020-xxt\»­Í¼\Cut_left.jpg');
imwrite(new_imgR,'C:\Users\lay\Desktop\ICME2020-xxt\»­Í¼\Cut_right.jpg');
[new_h, new_w, ~] = size(new_imgL);
disp('####')
disp(new_w)
disp(wRatio*w)

Ratio = (Ratio*w)/(new_w);
imL = new_imgL;
imR = new_imgR;
h = new_h;
w = new_w;


% the regular mesh on original image
Vertex_set_org = ImgRegualrMeshGrid(imL, mesh_size);
Vertex_set_org2 = Vertex_set_org;
k = (1-0.05)/(max(importance_map(:))-min(importance_map(:)));
importance_map = 0.05+k*(importance_map-min(importance_map(:)));
k = (1-0.05)/(max(importance_mapR(:))-min(importance_mapR(:)));
importance_mapR = 0.05+k*(importance_mapR-min(importance_mapR(:)));

importance_quad = SNS_importanceMap_quad(importance_map, Vertex_set_org);
importance_quad = importance_quad/sum(importance_quad(:)); % the importance weight for the quad
importance_quadR = SNS_importanceMap_quad(importance_mapR, Vertex_set_org2);
importance_quadR = importance_quadR/sum(importance_quadR(:)); % the importance weight for the quad
tic;
% generate sift feature
imL_Gray = rgb2gray(imL);
imwrite(imL_Gray,'GrayL.jpg');
imR_Gray = rgb2gray(imR);
imwrite(imR_Gray,'GrayR.jpg');
alpha_matrix = zeros(1,5);
alpha_matrixR = zeros(1,5);
if ALIGIN_TERM
    [p1,p2,num] = match('GrayL.jpg','GrayR.jpg',importance_map, 0);
    alpha_matrix = zeros(num,5);
    alpha_matrixR = zeros(num,5);


    for ii=1:num
       pL = [p1(ii,2) p1(ii,1)];
       v4 = [ceil(pL(1)-mod(floor(pL(1)),mesh_size)), ceil(pL(2)-mod(floor(pL(2)),mesh_size) )];
       if pL(1)<v4(1)
          v4 = [v4(1)-mesh_size v4(2)];
       end
       if pL(2)<v4(2)
          v4 = [v4(1) v4(2)-mesh_size];
       end
       v3 = [v4(1)+mesh_size v4(2)];
       v2 = [v4(1)+mesh_size v4(2)+mesh_size];
       v1 = [v4(1) v4(2)+mesh_size];
       alpha=compute_barycentric(pL,v1,v2,v3,v4);
       for temp=1:4
          if alpha(temp)<0
              break;
          end
          alpha_matrix(ii,temp) = alpha(temp); 

       end
       %alpha_matrix(ii,5) = ceil(quad_num_h*v4(1)/mesh_size+v4(2)/mesh_size);
       alpha_matrix(ii,5) = ceil(quad_num_h*(v4(1)-1)/mesh_size+(v4(2)-1)/mesh_size+1);
       pR = [p2(ii,2) p2(ii,1)];
       v4 = [ceil(pR(1)-mod(floor(pR(1)),mesh_size)), ceil(pR(2)-mod(floor(pR(2)),mesh_size) )];
       if pR(1)<v4(1)
          v4 = [v4(1)-mesh_size v4(2)];
       end
       if pR(2)<v4(2)
          v4 = [v4(1) v4(2)-mesh_size];
       end
       v3 = [v4(1)+mesh_size v4(2)];
       v2 = [v4(1)+mesh_size v4(2)+mesh_size];
       v1 = [v4(1) v4(2)+mesh_size];
       %alpha_matrixR(ii)=compute_barycentric(pR,v1,v2,v3,v4);
       alpha=compute_barycentric(pR,v1,v2,v3,v4);
       for temp=1:4
           if alpha(temp)<0
             %disp('############')
             %disp(v1)
             %disp(v2)
             %disp(v3)
             %disp(v4)
             %disp(pR)
             break;
          end
          alpha_matrixR(ii,temp) = alpha(temp); 

       end
       %alpha_matrixR(ii,5) = ceil(quad_num_h*v4(1)/mesh_size+v4(2)/mesh_size);
       alpha_matrixR(ii,5) = ceil(quad_num_h*(v4(1)-1)/mesh_size+(v4(2)-1)/mesh_size+1);
       if ii ==12
           disp('haha')
           disp(pL)
           disp(pR)
       end
    end
    alpha_matrix(:,1:4)=1/num*alpha_matrix(:,1:4);
    alpha_matrixR(:,1:4)=1/num*alpha_matrixR(:,1:4);
    
end
% the naive initialization of the mesh
% retargeting on the width
Vertex_warped_initial = Vertex_set_org;
Vertex_warped_initial(:,:,2) = Vertex_warped_initial(:,:,2)*Ratio;

% the mesh grid optimization
[Vertex_updated,Vertex_updatedR] = ...
    SNS_optimization(Vertex_set_org ,Vertex_warped_initial, importance_quad,importance_quadR, alpha_matrix, alpha_matrixR);

% warp the new image
im_warped = MeshBasedImageWarp(imL, [1 Ratio], Vertex_set_org, Vertex_updated);
im_warpedR = MeshBasedImageWarp(imR, [1 Ratio], Vertex_set_org2, Vertex_updatedR);
plot_mesh(imL,Vertex_set_org, alpha_matrix, 1);
plot_mesh(imR, Vertex_set_org, alpha_matrix, 1);
plot_mesh(im_warped,Vertex_updated, alpha_matrix, 0);
plot_mesh(im_warpedR, Vertex_updatedR, alpha_matrix, 0);
total_time = toc;
disp(['total time : ' num2str(total_time) ' s.']);
figure; subplot(2,2,1); imshow(im_warped); title(['My warped'], 'FontSize' , 15); 
subplot(2,2,2); imshow(im_warpedR); title(['My warpedR'], 'FontSize' , 15); 
%subplot(1,2,2); imshow(im_SNS); title(['Original SNS warped'], 'FontSize' , 15); 
subplot(2,2,3); imshow(imL); title(['Original image'], 'FontSize' , 15); 

%I_imerge = gen_red_cyn(im_warped,im_warpedR);
%subplot(2,1,2); imshow(I_imerge); title(['red cyn image'], 'FontSize' , 15); 
imwrite(im_warped,'my_warp_06-s.png');
imwrite(im_warpedR,'my_warpR_06-s.png');
%imwrite(I_imerge,'my_red_cyn_06-s.jpg');
saveas(gca,['fig' '.png']);
% show the mesh grid on the original image and retargeted image
% MeshGridImgPlot(im, Vertex_set_org, [0.5 0.0 0.5]);
% title(['Regular mesh grid on original image'], 'FontSize' , 15);
% MeshGridImgPlot(im_warped, Vertex_updated, [0.5 0.0 0.5]);
% title(['Warped image '], 'FontSize' , 15); 




