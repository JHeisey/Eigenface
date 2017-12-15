function [eigvec_svd,eigval_svd] = pca_svd(Xnorm,k)
%Function returns the eigenvectors of the covariance matrix for the given
%set. Essentially computes the eigenfaces in this application.
%

[U,S,V] = svd(Xnorm/sqrt(10304),'econ');
%Acquire only the first k eigenvectors of matrix
eigvec_svd = Xnorm*V;%Can swap out with U*S
eigvec_svd = eigvec_svd(:,1:k);
eigvec_svd = normc(eigvec_svd);
%Must square eigenvalues from the decomposition of the data matrix to get
%appropriate covariance matrix eigenvalues
eigval_svd = S.^2;

end