% num = match(image1, image2)
%
% This function reads two images, finds their SIFT features, and
%   displays lines connecting the matched keypoints.  A match is accepted
%   only if its distance is less than distRatio times the distance to the
%   second closest match.
% It returns the number of matches displayed.
%
% Example: match('scene.pgm','book.pgm');

function [p1,p2,num] = match(image1, image2,importance_map, isPlot)

% Find SIFT keypoints for each image
[im1, des1, loc1] = sift(image1);
[im2, des2, loc2] = sift(image2);
p1 = [0,0];
p2 = [0,0];
% For efficiency in Matlab, it is cheaper to compute dot products between
%  unit vectors rather than Euclidean distances.  Note that the ratio of 
%  angles (acos of dot products of unit vectors) is a close approximation
%  to the ratio of Euclidean distances for small angles.
%
% distRatio: Only keep matches in which the ratio of vector angles from the
%   nearest to second nearest neighbor is less than distRatio.
distRatio = 0.6;   

% For each descriptor in the first image, select its match to second image.
des2t = des2';                          % Precompute matrix transpose
for i = 1 : size(des1,1)
   dotprods = des1(i,:) * des2t;        % Computes vector of dot products
   [vals,indx] = sort(acos(dotprods));  % Take inverse cosine and sort results

   % Check if nearest neighbor has angle less than distRatio times 2nd.
   if (vals(1) < distRatio * vals(2))
      match(i) = indx(1);
   else
      match(i) = 0;
   end
end



% Show a figure with lines joining the accepted matches.
if isPlot
    % Create a new image showing the two images side by side.
    im3 = appendimages(im1,im2);
    figure('Position', [100 100 size(im3,2) size(im3,1)]);
    colormap('gray');
    imagesc(im3);
    hold on;

    cols1 = size(im1,2);
    cols2 = size(im1,1);
    %p1 = loc1(:,1:2);
    %p2 = p1;
    indd = 1;
    for i = 1: size(des1,1)
      if (match(i) > 0)
%         if abs(loc1(i,1)-loc2(match(i),1))>cols2*0.03 || importance_map(round(loc1(i,1)),round(loc1(i,2)))<0.2 || (i>1&&loc1(i,2)==loc1(i-1,2))
%         if abs(loc1(i,1)-loc2(match(i),1))>cols2*0.03 
         if abs(loc1(i,1)-loc2(match(i),1))>cols2*0.03 || (i>1&&loc1(i,2)==loc1(i-1,2))
             %disp(indd)
             %disp('error')
             line([loc1(i,2) loc2(match(i),2)+cols1], ...
             [loc1(i,1) loc2(match(i),1)], 'Color', 'red');
             continue;
        end
        if indd ==12
             disp('haha')
         end
        line([loc1(i,2) loc2(match(i),2)+cols1], ...
             [loc1(i,1) loc2(match(i),1)], 'Color', 'c');
         
         p1(indd,2) = loc1(i,2);
         p1(indd,1) = loc1(i,1);
         p2(indd,2) = loc2(match(i),2);
         p2(indd,1) = loc2(match(i),1);
         disp('point:y')
         disp(p1(indd,1))
         disp(p2(indd,1))
         indd = indd+1;
      end
    end

    hold off;
else
    %cols1 = size(im1,2);
    %p1 = loc1(:,1:2);
    %p2 = p1;
    cols2 = size(im1,1);
    indd = 1;
    for i = 1: size(des1,1)
      if (match(i) > 0)
         if abs(loc1(i,1)-loc2(match(i),1))>cols2*0.03 || importance_map(round(loc1(i,1)),round(loc1(i,2)))<0.1|| (i>1&&loc1(i,2)==loc1(i-1,2))
             %disp(indd)
             %disp('error')
             continue;
         end
         p1(indd,2) = loc1(i,2);
         p1(indd,1) = loc1(i,1);
         p2(indd,2) = loc2(match(i),2);
         p2(indd,1) = loc2(match(i),1);
         %disp('point:y')
         %disp(p1(indd,1))
         %disp(p2(indd,1))
         indd = indd+1;
      end
    end
end
%num = sum(match > 0);
num = indd-1;
fprintf('Found %d matches.\n', num);






