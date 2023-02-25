function [net, info] = vl_spdnet( covD_Struct, opts )

%#ok<*UNRCH,*NODEF,*AGROW,*NASGU,*SEPEX>
sigmaVal = covD_Struct.sigmaVal;
noiseVal = covD_Struct.noiseVal;
NetName = 'SPD-Net';
disp(['test on ', opts.experment]);
if strcmp(opts.experment, 'Toy-data')
    Problem = sprintf('toydata-%s-%s', num2str(sigmaVal), num2str(noiseVal));
    Problem_Tex = sprintf('Toy-data($\\sigma=%s,\\delta=%s$)',...
        num2str(sigmaVal),num2str(noiseVal));
    dataDir = fullfile('./data/Toy-data') ;
    figDir = fullfile('./data/Toy-data/figure') ;
    opts.dataDir = fullfile(dataDir, opts.NetName, sprintf('result-%s', Problem)) ;
    modelFigPath = fullfile(figDir, sprintf('%s-train-%s', opts.NetName, Problem)) ;
elseif strcmp(opts.experment, 'EEG-data')
    Problem = sprintf('EEGdata-%s-%s', num2str(sigmaVal), num2str(noiseVal));
    Problem_Tex = sprintf('EEG-data($\\sigma=%s,\\delta=%s$)',...
        num2str(sigmaVal),num2str(noiseVal));
    dataDir = fullfile('./data/EEG-data/SPD-Mani-Net') ;
    opts.dataDir = fullfile(dataDir, sprintf('result-eegdata-%1.0f', covD_Struct.subject)) ;
    modelFigPath = fullfile(opts.dataDir, sprintf('net-train-%1.0f', covD_Struct.subject)) ;
end
if ~exist(opts.dataDir, 'dir') && opts.savemodel
    mkdir(opts.dataDir)
end
if ~exist(figDir, 'dir') && opts.savefig
    mkdir(figDir)
end

%% spdnet initialization
rng('default') ;
rng(0) ;
Winit = cell(opts.layernum, 1) ;
for iw = 1 : opts.layernum
    A = rand(opts.datadim(iw)) ;
    [U1, ~, ~] = svd(A * A') ;
    Winit{iw} = U1(:, 1:opts.datadim(iw+1)) ;
end

f = 1/100 ;
classNum = covD_Struct.nClasses;
fdim = size(Winit{iw},2)*size(Winit{iw},2);
theta = f*randn(fdim, classNum, 'single');
Winit{end+1} = theta;
net.NetName = NetName;
net.layers = {} ;
net.layers{end+1} = struct('type', 'bfc', 'weight', Winit{1}) ;
net.layers{end+1} = struct('type', 'rec','epsilon', 1e-4) ;
net.layers{end+1} = struct('type', 'bfc', 'weight', Winit{2}) ;
net.layers{end+1} = struct('type', 'rec','epsilon', 1e-4) ;
net.layers{end+1} = struct('type', 'bfc', 'weight', Winit{3}) ;
net.layers{end+1} = struct('type', 'log') ;
net.layers{end+1} = struct('type', 'fc', 'weight', Winit{end}) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;

%% delete net-epoch
modelPath = @(ep) fullfile(opts.dataDir, sprintf('net-epoch-%.*d.mat', floor(log10(opts.numEpochs))+1, ep)) ;
if opts.clear
    for delete_epoch = 1 : opts.numEpochs
        if exist(modelPath(delete_epoch), 'file')
            delete(modelPath((delete_epoch)));
        else
            continue;
        end
    end
end

%%
opts.errorLabels = {'top1e'};
opts.train = find(covD_Struct.set==1) ;
opts.val = find(covD_Struct.set==2) ;
trn_y = covD_Struct.label(opts.train);
tst_y = covD_Struct.label(opts.val);


for epoch = 1 : opts.numEpochs
    Time = tic ;
    learningRate = opts.learningRate(epoch) ;
    %% load
    if opts.continue
        if exist(modelPath(epoch), 'file')
            if epoch == opts.numEpochs
                load(modelPath(epoch), 'net', 'info') ;
            end
            continue ;
        end
        if epoch > 1 && exist(modelPath(epoch-1), 'file')
            fprintf('resuming by loading epoch %d\n', epoch-1) ;
            load(modelPath(epoch-1), 'net', 'info') ;
        end
    end

    %% process_epoch
    train = opts.train(randperm(length(opts.train))) ; % shuffle
    val = opts.val;

    [net, stats.train] = process_epoch(opts, epoch, covD_Struct.spd, covD_Struct.label, train, learningRate, net) ;
    [net, stats.val] = process_epoch(opts, epoch, covD_Struct.spd, covD_Struct.label, val, 0, net) ;
    epochTime = toc(Time) ;

    %% save
    info.train.objective(epoch) = stats.train(2) / length(trn_y) ;
    info.val.error(epoch) = stats.val(3) / length(tst_y) ;
    info.time(epoch) = epochTime;

    if opts.savemodel
        save(modelPath(epoch), 'net', 'info') ;
    end

end
savefig = figure ;
FontSize = 12;
set(gcf,'unit','centimeters','position',[10 5 14 8.6].*[1 1 1 1]);
set(gcf, 'DefaultAxesFontSize', FontSize);
out = tight_subplot(1, 2, [.05 .1], [.12 .25], [.1 .05]);
axes(out(1));
semilogy(1:epoch, info.train.objective, 'b.-', 'linewidth', 2) ;
xlabel('epoch') ; ylabel('energy') ; grid on ;
title('objective on train set') ;
ax = gca;
ax.FontName = 'Times New Roman';
axes(out(2));
plot(1:epoch, info.val.error', 'r.--') ;
xlabel('epoch') ; ylabel('error') ; grid on ;
title('error on test set') ;
ax = gca;
ax.FontName = 'Times New Roman';
expname=sprintf('%s\n %s: $\\rm{epoch}:%s/%s$', Problem_Tex,opts.NetName,num2str(epoch),num2str(opts.numEpochs));
sup = suptitle(expname);
sup.Interpreter='latex';
sup.FontSize=16;
sup.Position=[0.5,-0.1,0];
saveas(savefig, sprintf('%s.%s', modelFigPath,'eps'), 'epsc');
print(savefig,'-dpng',sprintf('%s.%s', modelFigPath,'png'));

end

%% process_epoch
function [net, stats] = process_epoch(opts, epoch, spd_data, spd_label, trainInd, learningRate, net) %#ok<INUSL>
training = learningRate > 0 ;
if training
    mode = 'training' ;
else
    mode = 'validation' ;
end
stats = [0 ; 0 ; 0] ;
numGpus = numel(opts.gpus) ;
if numGpus >= 1
    one = gpuArray(single(1)) ;
else
    one = single(1) ;
end
batchSize = opts.batchSize ;
errors = 0;
numDone = 0 ;
for ib = 1 : batchSize : length(trainInd)
    batchTime = tic ;
    res = [] ;
    if (ib+batchSize> length(trainInd))
        batchSize_r = length(trainInd)-ib+1 ;
    else
        batchSize_r = batchSize ;
    end
    spd.data = cell(batchSize_r, 1) ;
    spd.label = zeros(batchSize_r, 1) ;
    for ib_r = 1 : batchSize_r
        spd.data{ib_r} = spd_data(:,:,trainInd(ib+ib_r-1)) ;
        spd.label(ib_r) =  spd_label(trainInd(ib+ib_r-1)) ;
    end
    net.layers{end}.class = spd.label ;
    if training, dzdy = one; else dzdy = [] ; end
    res = vl_myforbackward(net, spd.data, dzdy, res) ;

    if numGpus <= 1
        [net, res] = accumulate_gradients(opts, learningRate, batchSize_r, net, res) ;
    else
        if isempty(mmap)
            mmap = map_gradients(opts.memoryMapFile, net, res, numGpus) ;
        end
        write_gradients(mmap, net, res) ;
        labBarrier() ;
        [net, res] = accumulate_gradients(opts, learningRate, batchSize_r, net, res, mmap) ;
    end

    predictions = gather(res(end-1).x) ;
    [~,pre_label] = sort(predictions, 'descend') ;
    error = sum(~bsxfun(@eq, pre_label(1,:)', spd.label)) ;

    numDone = numDone + batchSize_r ;
    errors = errors+error;
    batchTime = toc(batchTime) ;
    stats = stats+[batchTime ; res(end).x ; error]; % works even when stats=[]
end

end

%% accumulate_gradients
function [net, res] = accumulate_gradients(opts, lr, batchSize, net, res, mmap) %#ok<INUSD,INUSL>
for l=numel(net.layers):-1:1
    if isempty(res(l).dzdw)==0
        if ~isfield(net.layers{l}, 'learningRate')
            net.layers{l}.learningRate = 1 ;
        end
        if ~isfield(net.layers{l}, 'weightDecay')
            net.layers{l}.weightDecay = 1 ;
        end
        thisLR = lr * net.layers{l}.learningRate ;
        if isfield(net.layers{l}, 'weight')
            if strcmp(net.layers{l}.type, 'bfc') == 1
                W1=net.layers{l}.weight;
                W1grad = (1/batchSize)*res(l).dzdw{1};
                problemW1.M = stiefelfactory(size(W1,1), size(W1,2));
                W1Rgrad = (problemW1.M.egrad2rgrad(W1, W1grad));
                net.layers{l}.weight = (problemW1.M.retr(W1, -thisLR*W1Rgrad)); %%!!!NOTE
            else
                net.layers{l}.weight = net.layers{l}.weight - thisLR * (1/batchSize)* res(l).dzdw ;
            end
        end
    end
end

end

