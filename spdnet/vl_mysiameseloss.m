function [ y1, y2, margin ] = vl_mysiameseloss( x1, x2, c, dzdy, margin_ )
% contrastive loss
%   L(D, L) = sum(L * D^2 + (1-L) * max(M - D, 0)^2) as defined in [1].
%
%   [DZDX1, DZDX2] = VL_NNCONTRLOSS(X1, X2, C, DZDY) computes the
%   derivative of the block projected onto the output derivative DZDY.
%   DZDX1, DZDX2 and DZDY have the same dimensions as X1, X2 and Y
%   respectively.
%
%   [1] Hadsell, Raia, Sumit Chopra, and Yann LeCun. "Dimensionality
%   reduction by learning an invariant mapping." CVPR 2006

% x1, x2格式为 K*K*N为矩阵，储存N对K维SPD矩阵
% c的格式为1*1*N矩阵，储存对应的标签

% margin = 5 ;
nel = length(x1) ;
size_ = size(x1{1}) ;
dist2 = zeros(nel, 1) ;
% a = sqrt(sum(log(eig(A,B)).^2));
for k = 1 : nel
    dist2(k) = sum(log(eig(x1{k}, x2{k})).^2) ;
end
dist = sqrt(dist2) ;
% margin_ = mean(dist);
% % % margin_ = median(dist); 
% % % if isempty(dzdy)
% % %     if margin == 0
% % %         margin = margin_;
% % %     elseif margin ~= 0
% % % %         margin = min(margin, margin_)-0.001*randn(1);
% % %         margin = min(margin, margin_);
% % %     end
% % % end
% margin = 10 ;
margin = max(margin_, prctile(dist, 75));%margin
% margin = min(margin_, mean(dist));%margin

mdist = margin - dist ;
% c == 0 -> 异类, c == 1 -> 同类
if nargin < 4 || isempty(dzdy)
    % 正向输出
    dist2(c == 0) = max(mdist(c == 0), 0).^2 ;
    y1 = sum(dist2) ; y2 = [];
else
    % 反向输出
    %     y1 = zeros(size(x1)) ; y2 = zeros(size(x2)) ;
    y1 = cell(nel, 1) ; y2 = cell(nel, 1) ; I_r = eye(size_, 'single') ;
    one = single(1) ;
    mdist = squeeze(mdist) ;
    nf = mdist ./ (dist + 1e-4*one) ;
    
    for k = 1 : nel
        if c(k) ==1
            log_XY_INV = logm(x1{k} * I_r/x2{k}) ;
            y1{k} = dzdy * I_r/x1{k} * log_XY_INV ;
            y2{k} = -dzdy * I_r/x2{k} * log_XY_INV ;
        elseif  c(k) ==0 && mdist(k) > 0
            log_XY_INV = logm(x1{k} * I_r/x2{k}) ;
            y1{k} = (-dzdy*nf(k))*I_r/x1{k} * log_XY_INV ;
            y2{k} = (dzdy*nf(k))*I_r/x2{k} * log_XY_INV ;
        elseif c(k) ==0 && mdist(k) <= 0
            y1{k} = zeros(size_, 'single') ;
            y2{k} = zeros(size_, 'single') ;
        end
    end
    margin = [];
end

end
%     nfcell = mat2cell(nf, ones(size(d, 2), 1), 1) ;
%     cellfun(@times, b{c==1}, b{aa==1})
%     neg_sel = mdist >  0 & c == 0 ;
%     tol_sel = mdist > 0 ;
%     y1{neg_sel} = bsxfun(@times, -dzdy, bsxfun(@times, y1{neg_sel}, nf(neg_sel))) ;
%     y2{neg_sel} = bsxfun(@times, -dzdy, bsxfun(@times, y2{neg_sel}, nf(neg_sel))) ;
%     y1{neg_sel} = bsxfun(@times, -y1{neg_sel}, nf(neg_sel)) ;
%     y2{neg_sel} = bsxfun(@times, -y2{neg_sel}, nf(neg_sel)) ;