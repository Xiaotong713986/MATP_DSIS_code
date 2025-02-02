clc
clear
addpath( genpath( '.' ) );
options.valScale = 60;
options.salThreshold = 0.2;
options.alpha = 0.002;
options.topRate = 0.01;
left = 1;
right = 2;
width_ratio = 0.5;

% set [w*h] of thumbnail via CPC
options.height = 100;
options.width = 50;


out_dir = '.\\output\\';

img_left_dir = '..\\examples\\origin-images\\left';
img_right_dir = '..\\examples\\origin-images\\right';
ImgsFiles{left} = fullfile(img_left_dir);
ImgsFiles{right} = fullfile(img_right_dir);

sal_left_dir = '..\\examples\\saliency-maps\\left';
sal_right_dir = '..\\examples\\saliency-maps\\right';
SalFiles{left} = fullfile(sal_left_dir);
SalFiles{right} = fullfile(sal_right_dir);

Imgs{left} = imdir(ImgsFiles{left});
Imgs{right} = imdir(ImgsFiles{right});
nimgs = length( Imgs{left});

for index = 1: nimgs       
        index;
        [~, ImgName{left}] = fileparts(Imgs{left}(index).name);
        [~, ImgName{right}] = fileparts(Imgs{right}(index).name);
        
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
        
        outfolder = fullfile(out_dir, ImgName{left});
        disp(outfolder)
        if( ~exist( outfolder, 'dir' ) ), mkdir( outfolder ), end;
        
        if( ~exist( fullfile( outfolder, 'regions'), 'dir' ) ), mkdir( fullfile( outfolder, 'regions') ), end;
        
        [ height,width ] = size(Img{left}(:,:,1));
        options.height = height;
        options.width = floor(width*width_ratio);
        
        PixNum = height*width;
        
        
        %% read saliency map
        
        if exist(fullfile(SalFiles{left}, [ImgName{left} '.png']),'file')
            Sal{left} = double(imread(fullfile(SalFiles{left}, [ImgName{left} '.png'])));
            Sal{right} = double(imread(fullfile(SalFiles{right}, [ImgName{right} '.png'])));
        elseif exist(fullfile(SalFiles{left}, [ImgName{left} '.jpg']),'file')
            Sal{left} = double(imread(fullfile(SalFiles{left}, [ImgName{left} '.jpg'])));
            Sal{right} = double(imread(fullfile(SalFiles{right}, [ImgName{right} '.jpg'])));
        elseif exist(fullfile(SalFiles{left}, [ImgName{left} '.bmp']),'file')
            Sal{left} = double(imread(fullfile(SalFiles{left}, [ImgName{left} '.bmp'])));
            Sal{right} = double(imread(fullfile(SalFiles{right}, [ImgName{right} '.bmp'])));
        end   
        
        lll = size(Sal{left});
        if size(lll)==3
            Sal{left} = rgb2gray(Sal{left});
            Sal{right} = rgb2gray(Sal{right});
        end
        
        sall = Sal{left};
        Sal{left} = (sall-min(sall(:)))/(max(sall(:))-min(sall(:)));  
        sall = Sal{right};
        Sal{right} = (sall-min(sall(:)))/(max(sall(:))-min(sall(:))); 
        
        
         %% compute superpixel
         Attr=[ height ,width, 1000, 20, PixNum ];
         imgVecR = reshape( Img{left}(:,:,1)', PixNum, 1);
         imgVecG = reshape( Img{left}(:,:,2)', PixNum, 1);
         imgVecB = reshape( Img{left}(:,:,3)', PixNum, 1); 
         disp('compute superpixel of left image:')

         [ leftLabel, leftSup1, leftSup2, leftSup3, RegionNum{left} ] = SLIC( double(imgVecR), double(imgVecG), double(imgVecB), Attr );
         
         superpixels{left}.Label = int32(reshape(leftLabel+1,width,height)');
         superpixels{left}.Lab = [leftSup1 leftSup2 leftSup3] ;
         superP{1} = uint32(superpixels{left}.Label);
         
         [ superpixels{left}.colours, superpixels{left}.centres, superpixels{left}.t ] = getSuperpixelStats(Img(left:left), superP, RegionNum{left} );         
         
         subImg{left} = superpixel2pixel(double(superpixels{left}.Label),double(superpixels{left}.colours));
         subImg{left} = uint8(reshape(subImg{left},height,width,3));

         imgVecR = reshape( Img{right}(:,:,1)', PixNum, 1);
         imgVecG = reshape( Img{right}(:,:,2)', PixNum, 1);
         imgVecB = reshape( Img{right}(:,:,3)', PixNum, 1); 
         
         disp('compute superpixel of right image:')

         [ rightLabel, rightSup1, rightSup2, rightSup3, RegionNum{right} ] = SLIC( double(imgVecR), double(imgVecG), double(imgVecB), Attr );
         
         superpixels{right}.Label = int32(reshape(rightLabel+1,width,height)');
         superpixels{right}.Lab = [rightSup1 rightSup2 rightSup3] ;
         superP{1} = uint32(superpixels{right}.Label);
         
         [ superpixels{right}.colours, superpixels{right}.centres, superpixels{right}.t ] = getSuperpixelStats(Img(right:right), superP, RegionNum{right} ); 
         
         subImg{right} = superpixel2pixel(double(superpixels{right}.Label),double(superpixels{right}.colours));
         subImg{right} = uint8(reshape(subImg{right},height,width,3));

         %% considering center-bais
         disp('center-bais of left image')

        nLabel = double(max(superpixels{left}.Label(:)));
        L{1} = uint32(superpixels{left}.Label);
        sall = Sal{left};
        S{1} = repmat(sall(:),[1,3]);
        [ R, ~, ~ ] = getSuperpixelStats(S(1:1),L, nLabel );
        R = double(R(:,1));
        R = (R-min(R))/(max(R)-min(R));
        
        IniSal_region{left} = R;
        IniSal{left} = double(R(superpixels{left}.Label));
        [IniSal_region{left} IniSal{left} mask] = centerBias(IniSal_region{left},IniSal{left},superpixels{left},options);
        
        disp('center-bais of right image')

        nLabel = double(max(superpixels{right}.Label(:)));
        L{1} = uint32(superpixels{right}.Label);
        sall = Sal{right};
        S{1} = repmat(sall(:),[1 3]);
        [ R, ~, ~ ] = getSuperpixelStats(S(1:1),L, nLabel );
        R = double(R(:,1));
        R = (R-min(R))/(max(R)-min(R));
        IniSal_region{right} = R;
        IniSal{right} = double(R(superpixels{right}.Label));
        [IniSal_region{right} IniSal{right} mask] = centerBias(IniSal_region{right},IniSal{right},superpixels{right},options);

        %% saliency optimization
        disp('saliency optimization')

        Sal = computeUS(IniSal_region,IniSal,superpixels,options);

        imwrite(Sal{left},[outfolder '/' ImgName{left}  '_sall.bmp']);
        imwrite(Sal{right},[outfolder '/' ImgName{right}  '_salr.bmp']);        
end
