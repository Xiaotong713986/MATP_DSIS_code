function [new_imgL,new_imgR,new_salL, new_salR] =cutImg3(imgL, imgR, salL, salR, wRatio, hRatio)
    %%%% 
    [h,w,~] = size(imgL);
    %hRatio=1;
    left = 1;
    right = w;
    threshold = 0.2;
    space = ceil(w*0.002);
    sal_value = zeros(1,w);
    left_flag = 1;
    right_flag = 1;
    
    
    if hRatio == 1
        for j=1:w
            sal_value(1,j)=sum(salL(:,j));
        end
        disp('cropping image.........')
        for i=1:w
            salL_v = sum(salL(:,i));
            salR_v = sum(salR(:,i));
            salL_v2 = sum(salL(:,w-i+1));
            salR_v2 = sum(salR(:,w-i+1));
            if (salL_v>threshold*max(sal_value) || salR_v>threshold*max(sal_value)) && left_flag
               left = max(1,i-space);     
               left_flag = 0;
            end
            if (salL_v2>threshold*max(sal_value) || salR_v2>threshold*max(sal_value)) && right_flag
               right = min(w,w-i+1+space);     
               right_flag = 0;
            end

            if left_flag == 0 && right_flag == 0
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
        imshow(salL)
        hold on;
        I = imgL;
        %I = cat(3,salL,salL,salL);
        I(:,left:left+4,:)=0;
        I(:,right:right+4,:)=0;
        %imwrite(I,'C:\Users\lay\Desktop\ICME2020-xxt\»­Í¼\Cut_left_.jpg');
        I = imgR;
        %I = cat(3,salL,salL,salL);
        I(:,left:left+4,:)=0;
        I(:,right:right+4,:)=0;
        %imwrite(I,'C:\Users\lay\Desktop\ICME2020-xxt\»­Í¼\Cut_right_.jpg');
        
    else
        threshold = 0.05;
        for j=1:h
            sal_value(1,j)=sum(salL(j,:));
        end
        disp('cropping image.........')
        for i=1:h
            salL_v = sum(salL(i,:));
            salR_v = sum(salR(i,:));
            salL_v2 = sum(salL(h-i+1, :));
            salR_v2 = sum(salR(h-i+1, :));
            if (salL_v>threshold*max(sal_value) || salR_v>threshold*max(sal_value)) && left_flag
               left = max(1,i-space);     
               left_flag = 0;
            end
            if (salL_v2>threshold*max(sal_value) || salR_v2>threshold*max(sal_value)) && right_flag
               right = min(h,h-i+1+space);     
               right_flag = 0;
            end

            if left_flag == 0 && right_flag == 0
                break;
            end
        end

        new_h = right-left;
        if new_h<hRatio*h
           temp = ceil((hRatio*h-new_h)*0.5);
           left = max(1,left-temp);
           right = min(h,right+temp);
        end
        new_imgL = imgL(left:right,:,:);
        new_imgR = imgR(left:right,:,:);
        new_salL = salL(left:right,:,:);
        new_salR = salR(left:right,:,:);
        imshow(salL)
        hold on;
%         I = imgL;
%         %I = cat(3,salL,salL,salL);
%         I(:,left:left+4,:)=0;
%         I(:,right:right+4,:)=0;
%         imwrite(I,'C:\Users\lay\Desktop\ICME2020-xxt\»­Í¼\Cut_left_.jpg');
%         I = imgR;
%         %I = cat(3,salL,salL,salL);
%         I(:,left:left+4,:)=0;
%         I(:,right:right+4,:)=0;
%         imwrite(I,'C:\Users\lay\Desktop\ICME2020-xxt\»­Í¼\Cut_right_.jpg');
    end
end