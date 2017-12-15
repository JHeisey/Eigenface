% This is a basic power method function which iteratively solves for the
% largest eigenvalue, it then performs the "method of delation" where it
% finds the largest eigenvalue and its associated eigenvector, then extends
% the covariance matrix to an orthogonal basis, where it can now find the
% second largest. This process continues until the desired number of
% largest eigenvalues/vectors are found.
%
%Inputs: Matrix A, initial guess vector V, maximum # of iterations,
%tolerance, k = desired number of eigenvalues found
%
%Outputs: list of k largest lambdas and column vector array of their
%associated eigenvectors.
%
function [Vs, lambdas] = pca_pow(A,V,it_max,tol,k)
%retain V for initial guess for each k iteration
V_init = V;
A_init = A;
A = cov(A);


%initialize empty array of eigvecs/vals
lambdas = zeros(1,length(k));
Vs = zeros(length(V),length(k));

%solve for first largest set using basic power method
[lambdas(1,1), Vs(:,1)] = power_method(A,V,it_max,tol);
%Begin iterative process of extending A to an orthogonal basis and
%repeating process
    for i = 2:k
        A = A - (lambdas(1,i-1)./norm(Vs(:,i-1)).^2).*(Vs(:,i-1)*Vs(:,i-1)');
        [lambdas(1,i), Vs(:,i)] = power_method(A,V_init,it_max,tol);
    end
    
    Vs = A_init*Vs;
    Vs = normc(Vs);
end



%Standard power method utilized for iterative search process
function [lambda,V] = power_method(A,V,it_max,tol)
  b = A*V;
  lambda = V'*b;
  V = b/norm(b);
  if (lambda < 0)
    V = -V;
  end
  %Only run power method iterations until maximum iterations is reached, or until
  %tolerance is reached.
  for iters = 1:it_max
    lambda_init = lambda;

    b = A*V;
    lambda = V'*b;
    V = b/norm(b);
    if (lambda < 0.0)
      V = -V;
    end

    val_dif = abs(lambda-lambda_init);

    if (val_dif <= tol)
      break
    end 
  end
  
  V = b/lambda;
end