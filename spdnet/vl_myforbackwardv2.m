function res = vl_myforbackwardv2(net, x, dzdy, res, varargin)
%#ok<*NASGU>
% vl_myforbackward  evaluates a simple SPDNet

opts.res = [] ;
opts.conserveMemory = false ;
opts.sync = false ;
opts.disableDropout = false ;
opts.freezeDropout = false ;
opts.accumulate = false ;
opts.cudnn = true ;
opts.skipForward = false;
opts.backPropDepth = +inf ;
opts.epsilon = 1e-4;

% opts = vl_argparse(opts, varargin);

n = numel(net.layers) ;

if (nargin <= 2) || isempty(dzdy)
    doder = false ;
else
    doder = true ;
end

if opts.cudnn
    cudnn = {'CuDNN'} ;
else
    cudnn = {'NoCuDNN'} ;
end

gpuMode = isa(x, 'gpuArray') ;

if nargin <= 3 || isempty(res)
    res = struct(...
        'x', cell(1,n+1), ...
        'dzdx', cell(1,n+1), ...
        'dzdw', cell(1,n+1), ...
        'aux', cell(1,n+1), ...
        'forwardTime', num2cell(zeros(1,n+1)), ...
        'backwardTime', num2cell(zeros(1,n+1))) ;
end
if ~opts.skipForward
    res(1).x = x ;
end


% -------------------------------------------------------------------------
% Forward pass
% -------------------------------------------------------------------------

for i = 1:n
    if opts.skipForward
        break;
    end
    l = net.layers{i} ;
    res(i).forwardTime = tic ;
    switch l.type
        case 'bfc'
            for k = 1:size(res(i).x, 2)
                res(i+1).x(:, k) = vl_mybfc(res(i).x(:, k), l.weight) ;
            end
        case 'shrinkage'
            for k = 1:size(res(i).x, 2)
                res(i+1).x(:, k) = vl_myshrinkage(res(i).x(:, k), l.epsilon) ;
            end
        case 'fc'
            res(i+1).x = vl_myfc(res(i).x, l.weight) ;
        case 'rec'
            for k = 1:size(res(i).x, 2)
                res(i+1).x(:, k) = vl_myrec(res(i).x(:, k), opts.epsilon) ;
            end
        case 'log'
            res(i+1).x = vl_mylog(res(i).x) ;
        case 'softmaxloss'
            res(i+1).x = vl_mysoftmaxloss(res(i).x, l.class) ;
        case 'siamese_loss'
            [res(i+1).x, ~, ~] =...%l.margin
                vl_mysiameseloss(res(i).x(:, 1), res(i).x(:, 2), l.label, [], l.margin) ;
        case 'custom'
            res(i+1) = l.forward(l, res(i), res(i+1)) ;
        otherwise
            error('Unknown layer type %s', l.type) ;
    end
    % optionally forget intermediate results
    forget = opts.conserveMemory ;
    forget = forget & (~doder || strcmp(l.type, 'relu')) ;
    forget = forget & ~(strcmp(l.type, 'loss') || strcmp(l.type, 'softmaxloss')) ;
    forget = forget & (~isfield(l, 'rememberOutput') || ~l.rememberOutput) ;
    if forget
        res(i).x = [] ;
    end
    if gpuMode && opts.sync
        % This should make things slower, but on MATLAB 2014a it is necessary
        % for any decent performance.
        wait(gpuDevice) ;
    end
    res(i).forwardTime = toc(res(i).forwardTime) ;
end

% -------------------------------------------------------------------------
% Backward pass
% -------------------------------------------------------------------------

if doder
    res(n+1).dzdx = dzdy ;
    for i=n:-1:max(1, n-opts.backPropDepth+1)
        l = net.layers{i} ;
        res(i).backwardTime = tic ;
        switch l.type
            case 'bfc'
                for k = 1:size(res(i).x, 2)
                    [res(i).dzdx(:, k), res(i).dzdw(:, k)] = ...
                        vl_mybfc(res(i).x(:, k), l.weight, res(i+1).dzdx(:, k)) ;
                end
                %                     [res(i).dzdx(:, 1), res(i).dzdw.l] = vl_mybfc(res(i).x(:, 1), l.weight, res(i+1).dzdx(:, 1)) ;
                %                     [res(i).dzdx(:, 2), res(i).dzdw.r] = vl_mybfc(res(i).x(:, 2), l.weight, res(i+1).dzdx(:, 2)) ;
            case 'shrinkage'
                for k = 1:size(res(i).x, 2)
                    res(i).dzdx(:, k) = ...
                        vl_myshrinkage(res(i).x(:, k), l.epsilon, res(i+1).dzdx(:, k)) ;
                end
            case 'fc'
                for k = 1:size(res(i).x, 2)
                    [res(i).dzdx(:, k), res(i).dzdw(:, k)]  = ...
                        vl_myfc(res(i).x(:, k), l.weight, res(i+1).dzdx(:, k)) ;
                end
            case 'rec'
                for k = 1:size(res(i).x, 2)
                    res(i).dzdx(:, k) = ...
                        vl_myrec(res(i).x(:, k), opts.epsilon, res(i+1).dzdx(:, k)) ;
                end
            case 'log'
                for k = 1:size(res(i).x, 2)
                    res(i).dzdx(:, k) = vl_mylog(res(i).x(:, k), res(i+1).dzdx(:, k)) ;
                end
            case 'softmaxloss'
                res(i).dzdx = vl_mysoftmaxloss(res(i).x, l.class, res(i+1).dzdx) ;
            case 'siamese_loss'
                res(i).dzdx = cell(length(res(i).x), size(res(i).x, 2)) ;
                [res(i).dzdx(:, 1), res(i).dzdx(:, 2), ~] =...
                    vl_mysiameseloss(res(i).x(:, 1), res(i).x(:, 2), l.label, res(i+1).dzdx, l.margin) ;
            case 'custom'
                res(i) = l.backward(l, res(i), res(i+1)) ;
        end
        if opts.conserveMemory
            res(i+1).dzdx = [] ;
        end
        if gpuMode && opts.sync
            wait(gpuDevice) ;
        end
        res(i).backwardTime = toc(res(i).backwardTime) ;
    end
end

