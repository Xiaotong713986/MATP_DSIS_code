function res=gen_red_cyn(imL,imR)
    imL_ = imL(:,:,2:3);
    imR_ = imR(:,:,1:2);
    res = cat(3,imL_,imR_);
end