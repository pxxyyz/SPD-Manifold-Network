function Y = vl_myshrinkage(X, epsilon, dzdy)
% Y = VL_MYREC (X, EPSILON, DZDY)
% ReEig layer

% Us = cell(length(X), 1) ;
% Ss = cell(length(X), 1) ;

D = size(X{1}, 2) ;
I = eye(D);
nu = zeros(1, D);
Y = cell(length(X), 1) ;

for ix = 1:length(X)
    %     [Us{ix}, Ss{ix}, ~] = svd(X{ix}) ;
    %     nu(ix) = trace(Ss{ix})/D;
    nu(ix) = trace(X{ix})/D;
    % nu(ix) = trace(X{ix});
end

if nargin < 3
    for ix = 1:length(X)
        Y{ix} = (1-epsilon)*X{ix}+epsilon*nu(ix)*I ;
    end
else
    for ix = 1:length(X)
        [U, S, ~] = svd(X{ix}) ;
        %         U = Us{ix} ; S = Ss{ix} ;
        Dmin = D ;

        dLdC = double(dzdy{ix}) ;
        dLdC = symmetric(dLdC) ;

        dLdV = 2*dLdC*U*((1-epsilon)*S + epsilon*nu(ix)*I) ;
        dLdS = (1-epsilon)*U'*dLdC*U ;

        K = 1./(diag(S)*ones(1, Dmin)-(diag(S)*ones(1, Dmin))') ;
        K(eye(size(K, 1))>0)=0 ;
        K(isinf(K)==1)=0 ;
        dzdx = U*(symmetric(K'.*(U'*dLdV))+dDiag(dLdS))*U' ;

        Y{ix} =  dzdx ; %warning('no normalization') ;
    end
end
