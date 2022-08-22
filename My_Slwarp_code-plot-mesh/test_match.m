[p1,p2,num] = match('0471_leftG.jpg','0471_rightG.jpg',0);
x1 = p1(:,2);
y1 = p1(:,1);
x2 = p2(:,2);
y2 = p2(:,1);

figure;
for i=1:num
    line([x1(i) x2(i)+960],[y1(i) y2(i)], 'Color', 'c');
end