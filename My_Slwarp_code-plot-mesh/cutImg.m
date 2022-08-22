function [new_imgL,new_imgR,new_salL, new_salR] =cutImg(imgL, imgR, salL, salR, wRatio, hRatio)
    [h,w,d] = size(imgL);
    hRatio=1;
    left = 1;
    right = w;
    space = ceil(w*0.002);
    
    for i=1:w
        salL_v = sum(salL(:,i));
        salR_v = sum(salR(:,i));
        if salL_v || salR_v
           left = max(1,i-space);     
           break;
        end
    end
    
    for j=1:w
       salL_v = sum(salL(:,w-j+1));
       salR_v = sum(salR(:,w-j+1));
       if salL_v || salR_v
           right = min(w,w-j+1+space);     
           break;
        end
    end
    
    new_w = right-left;
    if new_w<wRatio*w
       temp = ceil((wRatio*w-new_w)*0.5);
       left = max(1,left-temp);
       right = min(w,right+temp);
    end
    new_imgL = imgL(:,left:right,:);
    new_imgR = imgR(:,left:right,:);
    new_salL = salL(:,left:right,:);
    new_salR = salR(:,left:right,:);
end