function [ Vertex_updated, Vertex_updatedR ] = SNS_optimization(Vertex_set_org, Vertex_warped_initial, importance_quad,importance_quadR, alpha_matrix, alpha_matrixR)
% Summary of this function goes here
%   Detailed explanation goes here% The Code (Version 1) is created by ZHANG Yabin,
% Nanyang Technological University, 2015-12-30
% which is based on the method described in the following paper 
% [1] Wang, Yu-Shuen, et al. "Optimized scale-and-stretch for image resizing." 
% ACM Transactions on Graphics (TOG) 27.5 (2008): 118. 
% The binary code is provided on the project page:
% http://graphics.csie.ncku.edu.tw/Image_Resizing/
% The Matlab codes are for non-comercial use only.
% Note that the importance maps are slightly different from the original
% ones, and the retargeted images are influenced.

[Vertex_updated, Vertex_updatedR ] = ...
    SNS_Opt_Iter_M3(Vertex_set_org ,Vertex_warped_initial, Vertex_warped_initial,importance_quad,importance_quadR, alpha_matrix, alpha_matrixR);

Vertex_updated_old = Vertex_updated;

Vertex_max_move = inf;
Iter_NUM = 1;
while(Vertex_max_move > 0.5)
    Iter_NUM = Iter_NUM + 1;
    disp(['########## Iteration no. ' num2str(Iter_NUM)]);
    [Vertex_updated, Vertex_updatedR] = ...
        SNS_Opt_Iter_M3(Vertex_set_org ,Vertex_updated, Vertex_updatedR, importance_quad,importance_quadR, alpha_matrix, alpha_matrixR);
    Vertex_max_move = max(max(max(abs(Vertex_updated_old - Vertex_updated))));
    Vertex_updated_old = Vertex_updated;
    disp(['--- Max movement =  ' num2str(Vertex_max_move, '%.3f')]);
end

end

