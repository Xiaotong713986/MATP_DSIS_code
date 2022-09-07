left = 1;
right = 2;
Ratio = 1;
hRatio = 0.6;
ImgsFiles{left} = fullfile('..\\examples\\origin-images\\left\\');
ImgsFiles{right} = fullfile('..\\examples\\origin-images\\right\\');

SalFiles{left} = fullfile('..\\examples\\saliency-object-maps\\left\\');
SalFiles{right} = fullfile('..\\examples\\saliency-object-maps\\right\\');

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
    [im_warped, im_warpedR]=SLWAP(ImgName{left},Img{left}, Img{right}, Sal{left}, Sal{right}, Ratio, hRatio);

    save_dirL = '.\\output\\left\\';
    save_dirR = '.\\output\\right\\';
    %save_dir_fig = '.\\result_fig\\';
    imwrite(im_warped,[save_dirL, ImgName{left}, '.jpg']);
    imwrite(im_warpedR,[save_dirR, ImgName{right},'.jpg']);
    %saveas(gca,[save_dir_fig, imName]);
end
