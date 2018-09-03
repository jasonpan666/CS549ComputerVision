function K = compute_kernel(X1, X2,ind1,ind2,hp)
  switch hp.type
    
   case 'linear'
    if isempty(ind2)
      K = sum(X(ind1,:).^2,2);
      return;
    end;
    K = X(ind1,:)*X(ind2,:)';
    
   case 'rbf'
    if isempty(ind2)
      K = ones(length(ind1),1);
      return;
    end;
    normX = sum(X1(ind1,:).^2,2);
    normY = sum(X2(ind2,:).^2,2);
    K = exp(-0.5/hp.sig^2*(repmat(normX ,1,length(ind2)) + ...
                           repmat(normY',length(ind1),1) - ...
                           2*X1(ind1,:)*X2(ind2,:)'));
   otherwise
    error('Unknown kernel');
  end;