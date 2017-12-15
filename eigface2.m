clear;
%Initialize variables and constants
people=10;%Number of different people to select max 7 for training set
poses=7;%Number of poses per person
k = 60;%Dimensionality reduction
pick = 1;%Individual person picked to reconstruct
m = people*poses;%Number of images

%Reads in the data and splits it into a training and test set
[X,Xtest,r,c] = read_data(people,poses);

%mean of input faces
Xmean = mean(X,2);
%Subtract the mean from the faces to feature normalize
Xnorm = X - Xmean;
%Acquire the first k ordered eigenvectors and eigenvalues utilizing svd 
[eigvec_svd,eigval_svd] = PCA_QR(Xnorm,k);

[evectors, score, evalues] = pca(Xnorm');
evectors = evectors(:,1:k);
% display the eigenvectors
figure(1);hold on
for n = 1:9
    subplot(5, ceil(5/2), n);
    evector = mat2gray(reshape(evectors(:,n), [112 92]));
    imshow(evector);
end
hold off

% display the eigenvectors
figure(2);hold on
for n = 1:9
    subplot(4, ceil(5/2), n);
    evector = mat2gray(reshape(eigvec_svd(:,n), [112 92]));
    imshow(evector);
end
hold off

%Acquire k weights for each of the samples 
W = weights(Xnorm,evectors,k);
W_svd = weights(Xnorm,eigvec_svd,k);

%Plot the image of the person being reconstructed
figure(3);hold on
imshow(mat2gray(reshape(X(:,pick),[112 92])))
hold off


%Plots for reconstruction from weights
figure(5);hold on
recon = reconstruct(W(pick,:),Xmean,evectors);
imshow(mat2gray(reshape(recon(pick,:),[112 92])))
hold off

figure(6);hold on
recon_svd = reconstruct(W_svd(pick,:),Xmean,eigvec_svd);
imshow(mat2gray(reshape(recon_svd(pick,:),[112 92])))
hold off

%Generate the weights of the testing set
Xtestnorm = Xtest-Xmean;%normalize the test set

%Acquire the new projection of the test set 
W_test = weights(Xtestnorm,evectors,k);

%Compare the euclidean distances of the test set projection for the
%specific person selected
[mindist,person,dist] = match(W,W(1,:));

%Print both the search image and the closest training set image
figure(8);hold on
subplot(1,2,1)
imshow(mat2gray(reshape(Xtest(:,pick),[112 92])))
title('Test Image')
subplot(1,2,2)
imshow(mat2gray(reshape(X(:,person),[112 92])))
title('Match')
hold off




