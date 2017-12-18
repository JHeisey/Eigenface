%This is the QR function utilized in the PCA context
%
%Inputs: Matrix m, # of desired eigen vectors k, width of rectangular
%data matrix w, mean of matrix mu
%
%Outputs: Eigen Vector column matrix and diagonal Eigenvalue matrix
function [ VV, D1 ] = PCA_QR(m,k,w,mu)
%Find dimensions & mean of matrix
[w, h] = size(m);
mu = mean(m);
%normalize matrix
H = (1/sqrt(w))*(m-mu);

%perform built in QR method
[q1, r1] = qr(H,0);

% correlation matrix = q*r(q*r)'
% do svd on r' to get r' = udv'
% d^2 eigenvalues 
% qv eigenvectors

%perform 'economic' svd function
[u1, d1, v1] = svd(r1','econ');

%Extract desired k eigenvectors
%and eigenvalues
V = v1(1:k,:);
VV = q1*(V');
D1 = d1.^2;

end

