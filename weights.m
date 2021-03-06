function W = weights(Xnorm,eigvec,k)
%Computes the weight vector for each sample. Calculates the projection of a
%face in k dimenions to acquire eigenface representation.
%

W = Xnorm'*eigvec;

if k < size(eigvec,2)
    W = W(:,1:k);
end


end