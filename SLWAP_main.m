left = 1;
right = 2;
Ratio = 0.6;
hRatio = 1;
% NBU_SIRQA_Database
% ImgsFiles{left} = fullfile('D:\myData\2D3DSaliency\Feng_Shao_SIRQA\NBU_SIRQA_Database\all_result\origin\left\');
% ImgsFiles{right} = fullfile('D:\myData\2D3DSaliency\Feng_Shao_SIRQA\NBU_SIRQA_Database\all_result\origin\right\');
% SalFiles{left} = fullfile('D:\myData\2D3DSaliency\Feng_Shao_SIRQA\NBU_SIRQA_Database\NBU_SIRQA_sal_cpc05_1024\left\');
% SalFiles{right} = fullfile('D:\myData\2D3DSaliency\Feng_Shao_SIRQA\NBU_SIRQA_Database\NBU_SIRQA_sal_cpc05_1024\right\');

% S3D Database
% ImgsFiles{left} = fullfile('D:\myData\MyResearch\S3D-database\origin\left\');
% ImgsFiles{right} = fullfile('D:\myData\MyResearch\S3D-database\origin\right\');
% SalFiles{left} = fullfile('D:\myData\MyResearch\S3D-database\Our_sal_S3D\S3D_bias_dilate\left\');
% SalFiles{right} = fullfile('D:\myData\MyResearch\S3D-database\Our_sal_S3D\S3D_bias_dilate\right\');
ImgsFiles{left} = fullfile('D:\myData\MTAP_conference\DSIS_exp_result\S3D_database_res\Origin\left\');
ImgsFiles{right} = fullfile('D:\myData\MTAP_conference\DSIS_exp_result\S3D_database_res\Origin\right\');
SalFiles{left} = fullfile('D:\myData\MTAP_conference\DSIS_exp_result\S3D_database_res\Origin\Our_sal\left\');
SalFiles{right} = fullfile('D:\myData\MTAP_conference\DSIS_exp_result\S3D_database_res\Origin\Our_sal\right\');
save_dirL = 'D:\myData\MTAP_conference\DSIS_exp_result\S3D_database_res\Ours_w0.6_woDepthPreTerm\left\';
save_dirR = 'D:\myData\MTAP_conference\DSIS_exp_result\S3D_database_res\Ours_w0.6_woDepthPreTerm\right\';

Imgs{left} = dir(ImgsFiles{left});
Imgs{right} = dir(ImgsFiles{right});
Sals{left} = dir(SalFiles{left});
Sals{right} = dir(SalFiles{right});

nimgs = length( Imgs{left});

for index = 3: nimgs       
        index
        
        [~, Sal_ImgName{left}] = fileparts(Sals{left}(index).name);
        [~, Sal_ImgName{right}] = fileparts(Sals{left}(index).name);
        
        ImgName{left} = [Sal_ImgName{left}];
        ImgName{right} = [Sal_ImgName{right}];
        disp(ImgName{left})
        disp(ImgName{right})
        disp(Sal_ImgName{left})
        disp(Sal_ImgName{right})
        if exist(fullfile([save_dirL, ImgName{left}, '.jpg']), 'file')
            continue;
        end
        if exist(fullfile(ImgsFiles{left}, [ImgName{left} '.png']),'file')
            Img{left} = double(imread(fullfile(ImgsFiles{left}, [ImgName{left} '.png'])));
            Img{right} = double(imread(fullfile(ImgsFiles{right}, [ImgName{right} '.png'])));
        elseif exist(fullfile(ImgsFiles{left}, [ImgName{left} '.jpg']),'file')
            Img{left} = double(imread(fullfile(ImgsFiles{left}, [ImgName{left} '.jpg'])));
            Img{right} = double(imread(fullfile(ImgsFiles{right}, [ImgName{right} '.jpg'])));
        elseif exist(fullfile(ImgsFiles{left}, [ImgName{left} '.bmp']),'file')
            Img{left} = double(imread(fullfile(ImgsFiles{left}, [ImgName{left} '.bmp'])));
            Img{right} = double(imread(fullfile(ImgsFiles{right}, [ImgName{right} '.bmp'])));
        elseif exist(fullfile(ImgsFiles{left}, [ImgName{left} '_left.png']),'file')
            Img{left} = double(imread(fullfile(ImgsFiles{left}, [ImgName{left} '_left.png'])));
            Img{right} = double(imread(fullfile(ImgsFiles{right}, [ImgName{right} '_right.png'])));
        elseif exist(fullfile(ImgsFiles{left}, [ImgName{left} '_left.jpg']),'file')
            Img{left} = double(imread(fullfile(ImgsFiles{left}, [ImgName{left} '_left.jpg'])));
            Img{right} = double(imread(fullfile(ImgsFiles{right}, [ImgName{right} '_right.jpg'])));
        elseif exist(fullfile(ImgsFiles{left}, [ImgName{left} '_left.bmp']),'file')
            Img{left} = double(imread(fullfile(ImgsFiles{left}, [ImgName{left} '_left.bmp'])));
            Img{right} = double(imread(fullfile(ImgsFiles{right}, [ImgName{right} '_right.bmp'])));
        end;
        
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
        
        tic
        [im_warped, im_warpedR]=SLWAP(Img{left}, Img{right}, Sal{left}, Sal{right}, Ratio, hRatio);
        toc
        
        imwrite(im_warped,[save_dirL, ImgName{left}, '.jpg']);
        imwrite(im_warpedR,[save_dirR, ImgName{right},'.jpg']);
end
