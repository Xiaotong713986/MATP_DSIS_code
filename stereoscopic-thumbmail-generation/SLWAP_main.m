left = 1;
right = 2;
Ratio = 1;
hRatio = 0.6;
%ImgsFiles{left} = fullfile('C:\Users\lay\Desktop\2D3DSaliency\Feng_Shao_SIRQA\NBU_SIRQA_Database\all_result\origin\left\');
%ImgsFiles{right} = fullfile('C:\Users\lay\Desktop\2D3DSaliency\Feng_Shao_SIRQA\NBU_SIRQA_Database\all_result\origin\right\');
%ImgsFiles{left} = fullfile('C:\Users\lay\Desktop\ICME2020-xxt\exp_result\image\left\');
%ImgsFiles{right} = fullfile('C:\Users\lay\Desktop\ICME2020-xxt\exp_result\image\right');
ImgsFiles{left} = fullfile('/data/qiudan/3DSaliency_Program/Feng_Shao_SIRQA/NBU_SIRQA_Database/NBU_SIRQA_origin/left/');
ImgsFiles{right} = fullfile('/data/qiudan/3DSaliency_Program/Feng_Shao_SIRQA/NBU_SIRQA_Database/NBU_SIRQA_origin/right/');
%SalFiles{left} = fullfile('C:\Users\lay\Desktop\2D3DSaliency\Feng_Shao_SIRQA\NBU_SIRQA_Database\NBU_SIRQA_sal_cpc05_1024\left\');
%SalFiles{right} = fullfile('C:\Users\lay\Desktop\2D3DSaliency\Feng_Shao_SIRQA\NBU_SIRQA_Database\NBU_SIRQA_sal_cpc05_1024\right\');
%SalFiles{left} = fullfile('C:\Users\lay\Desktop\2D3DSaliency\Feng_Shao_SIRQA\NBU_SIRQA_Database\Wang-2017_sal\left\');
%SalFiles{right} = fullfile('C:\Users\lay\Desktop\2D3DSaliency\Feng_Shao_SIRQA\NBU_SIRQA_Database\Wang-2017_sal\right\');
SalFiles{left} = fullfile('/data/qiudan/3DSaliency_Program/Feng_Shao_SIRQA/NBU_SIRQA_Database/Wang-2017_sal/left');
SalFiles{right} = fullfile('/data/qiudan/3DSaliency_Program/Feng_Shao_SIRQA/NBU_SIRQA_Database/Wang-2017_sal/right');
% disp(ImgsFiles)
Imgs{left} = dir(ImgsFiles{left});
Imgs{right} = dir(ImgsFiles{right});

Sals{left} = dir(SalFiles{left});
Sals{right} = dir(SalFiles{right});

nimgs = length( Imgs{left});


for index = 3: nimgs       
        index
%         [~, ImgName{left}] = fileparts(Imgs{left}(index).name);
%         [~, ImgName{right}] = fileparts(Imgs{right}(index).name);
        
        [~, Sal_ImgName{left}] = fileparts(Sals{left}(index).name);
        [~, Sal_ImgName{right}] = fileparts(Sals{left}(index).name);
%         Sal_ImgName{left} = '11';
%         Sal_ImgName{right} = '11';
        %ImgName{left} = [Sal_ImgName{left},'_left'];
        %ImgName{right} = [Sal_ImgName{right},'_right'];
        ImgName{left} = [Sal_ImgName{left}];
        ImgName{right} = [Sal_ImgName{right}];
        disp(ImgName{left})
        disp(ImgName{right})
        disp(Sal_ImgName{left})
        disp(Sal_ImgName{right})
        %disp(ImgName{left})
        %ImgName{left} = 'previous_DSCF0471_combi';
        %ImgName{right} = 'previous_DSCF0471_combi';
        if exist(fullfile(ImgsFiles{left}, [ImgName{left} '.png']),'file')
            Img{left} = double(imread(fullfile(ImgsFiles{left}, [ImgName{left} '.png'])));
            Img{right} = double(imread(fullfile(ImgsFiles{right}, [ImgName{right} '.png'])));
        elseif exist(fullfile(ImgsFiles{left}, [ImgName{left} '.jpg']),'file')
            Img{left} = double(imread(fullfile(ImgsFiles{left}, [ImgName{left} '.jpg'])));
            Img{right} = double(imread(fullfile(ImgsFiles{right}, [ImgName{right} '.jpg'])));
        elseif exist(fullfile(ImgsFiles{left}, [ImgName{left} '.bmp']),'file')
            Img{left} = double(imread(fullfile(ImgsFiles{left}, [ImgName{left} '.bmp'])));
            Img{right} = double(imread(fullfile(ImgsFiles{right}, [ImgName{right} '.bmp'])));
        end 
        
        if exist(fullfile(SalFiles{left}, [Sal_ImgName{left} '.png']),'file')
            Sal{left} = double(imread(fullfile(SalFiles{left}, [Sal_ImgName{left} '.png'])));
            Sal{right} = double(imread(fullfile(SalFiles{right}, [Sal_ImgName{right} '.png'])));
        elseif exist(fullfile(SalFiles{left}, [Sal_ImgName{left} '.jpg']),'file')
            Sal{left} = double(imread(fullfile(SalFiles{left}, [Sal_ImgName{left} '.jpg'])));
            Sal{right} = double(imread(fullfile(SalFiles{right}, [Sal_ImgName{right} '.jpg'])));
        elseif exist(fullfile(SalFiles{left}, [Sal_ImgName{left} '.bmp']),'file')
            Sal{left} = double(imread(fullfile(SalFiles{left}, [Sal_ImgName{left} '.bmp'])));
            Sal{right} = double(imread(fullfile(SalFiles{right}, [Sal_ImgName{right} '.bmp'])));
        end 
        
        %outfolder = fullfile( foldername, 'results_S3D_bias_dilate', ImgName{left});
        %outfolder = fullfile( foldername, 'test', ImgName{left});
%         outfolder = fullfile('/data/qiudan/3DSaliency_Program/Cropping/SLWAP/');
%         disp(outfolder)
        tic
        [im_warped, im_warpedR]=SLWAP(ImgName{left},Img{left}, Img{right}, Sal{left}, Sal{right}, Ratio, hRatio);
        toc
        %save_dirL = 'C:\Users\lay\Desktop\ICME2020-xxt\exp_result\cpc_sal_h06\left\';
        %save_dirR = 'C:\Users\lay\Desktop\ICME2020-xxt\exp_result\cpc_sal_h06\right\';
        save_dirL = '/data/qiudan/3DSaliency_Program/Retargeting/My_Slwarp_code/res_h05/left/';
        save_dirR = '/data/qiudan/3DSaliency_Program/Retargeting/My_Slwarp_code/res_h05/right/';
        %save_dir_fig = 'result/fig/';
        imwrite(im_warped,[save_dirL, ImgName{left}, '.jpg']);
        imwrite(im_warpedR,[save_dirR, ImgName{right},'.jpg']);
        %saveas(gca,[save_dir_fig, imName]);
end
