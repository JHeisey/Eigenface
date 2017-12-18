function recon = reconstruct(W,Xmean,eigenvec)
%Reconstructs an image using their weights and the projection subspace
%
projection = W*eigenvec';
recon = projection+ Xmean;

end