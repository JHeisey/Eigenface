function [ VV, D1 ] = PCA_QR2(m,k,w,mu)
%   Detailed explanation goes here
%[w, h] = size(m);
%mu = mean(m);

%X = m - repmat(mu, w, 1);

H = (1/sqrt(w))*(m-mu);

% [q, r] = qr(H,0);
[q1, r1] = qr(H,0);
% correlation matrix = q*r(q*r)'
% do svd on r' to get r' = udv'
% d^2 eigenvalues 
% qv eigenvectors

% [u, d, v] = svd(r);
[~, d1, v1] = svd(r1',0);

% V = q*v;
% D = d.^2;


V1 = q1*(v1');
D1 = d1.^2;

% Need to add q1v_h to get eigenvalue matrix!!
% VV=V1;
VV = V1(:,1:k);
% VV = V1(:,1:10);

% DD = eig(H*H');
end

