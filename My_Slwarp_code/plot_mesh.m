function res_img = plot_mesh(img,point,alpha_matrix, flag)
    num = size(alpha_matrix,1);
    [h,w,~] = size(point);
    x = reshape(point(:,:,2),1,h*w);
    y = reshape(point(:,:,1),1,h*w);
    figure;
    hold on;
    %imagesc(img);
    imshow(img)
    for i = 1:w-1
        for j = 1:h-1
            
            quad_loc = j+(i-1)*h;
            index = find(alpha_matrix(:,5)==quad_loc);
%             if index
%                 point_x = alpha_matrix(index,1)*x(j+1+(i-1)*h)+alpha_matrix(index,2)*x(j+1+(i)*h)+alpha_matrix(index,3)*x(j+(i)*h)+alpha_matrix(index,4)*x(j+(i-1)*h);
%                 point_y = alpha_matrix(index,1)*y(j+1+(i-1)*h)+alpha_matrix(index,2)*y(j+1+(i)*h)+alpha_matrix(index,3)*y(j+(i)*h)+alpha_matrix(index,4)*y(j+(i-1)*h);
%                 point_x = point_x(1)*num;
%                 point_y = point_y(1)*num;
% %                 disp(point_x)
% %                 disp(point_y)
%                 scatter(point_x,point_y, 20, 'filled','g')
%             end
            
            line([x(j+(i-1)*h), x(j+1+(i-1)*h)],[y(j+(i-1)*h), y(j+1+(i-1)*h)],'Color','r','LineWidth',1)
            line([x(j+(i-1)*h), x(j+(i)*h)],[y(j+(i-1)*h), y(j+(i)*h)],'Color','r','LineWidth',1)
            %disp([j+(i-1)*h, j+(i-1)*h+1])
            hold on;
            
            
            
        end
    end
end