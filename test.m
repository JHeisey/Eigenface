clear all;
close all;
yalefaces = eigface();
[h,w] = size(yalefaces);
d = h*w;
% vectorize images
x = double(yalefaces);
%subtract mean
x=bsxfun(@minus, x, mean(x,2));
% calculate covariance
s = cov(x');
% obtain eigenvalue & eigenvector
[V,D] = eig(s);

% show 0th through 15th principal eigenvectors
eig0 = reshape(mean(x,2), [h,w]);
figure,subplot(4,4,1)
imagesc(eig0)
colormap gray
for i = 1:5
    subplot(4,4,i+1)
    imagesc(reshape(V(:,i),h,w))
end