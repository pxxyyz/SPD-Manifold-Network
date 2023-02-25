
function vl_siamese( covD_Struct, opts )

%% test toy data On Siamese Net
% clearvars -except x ;  clc ;
close all ;
%#ok<*UNRCH>
%#ok<*NODEF>
%#ok<*AGROW>
%#ok<*NASGU>

if isfield(covD_Struct,'SigmaVal') && isfield(covD_Struct,'noiseVal')
    disp(' test on Toy-data ');
    dataDir = fullfile('./data/Toy-data/Siamese-Net') ;
    opts.dataDir = fullfile(dataDir, sprintf('result-toydata-%1.0f-%1.0f',...
        10*opts.SigmaVal, 10*opts.noiseVal)) ;
    modelFigPath = fullfile(opts.dataDir, sprintf('net-train-%1.0f-%1.0f.pdf',...
        10*covD_Struct.SigmaVal, 10*covD_Struct.noiseVal)) ;
    title_ = ['sigma = ', num2str(covD_Struct.SigmaVal), ...
        ', noise = ', num2str(covD_Struct.noiseVal)];
elseif isfield(covD_Struct, 'subject')
    disp(' test on EEG-data ');
    dataDir = fullfile('./data/EEG-data/Siamese-Net') ;
    opts.dataDir = fullfile(dataDir, sprintf('result-eegdata-%1.0f', covD_Struct.subject)) ;
    modelFigPath = fullfile(opts.dataDir, sprintf('net-train-%1.0f.pdf', covD_Struct.subject)) ;
    title_ = ['subject = ', num2str(covD_Struct.subject)];
end
if ~exist(opts.dataDir, 'dir')
    mkdir(opts.dataDir)
end

%% spdnet initialization
rng('default') ;
rng(1) ;
Winit = cell(opts.layernum, 1) ;
for iw = 1 : opts.layernum
    A = rand(opts.datadim(iw)) ;
    [U1, ~, ~] = svd(A * A') ;
    Winit{iw} = U1(:, 1:opts.datadim(iw+1)) ;
end
net.layers = {} ;
net.layers{end+1} = struct('type', 'shrinkage','epsilon', 0.05) ;%0.01
net.layers{end+1} = struct('type', 'bfc', 'weight', Winit{1}) ;
net.layers{end+1} = struct('type', 'shrinkage','epsilon', 0.01) ;%0.01
net.layers{end+1} = struct('type', 'bfc', 'weight', Winit{2}) ;
net.layers{end+1} = struct('type', 'shrinkage','epsilon', 0.01) ;%0.005
% net.layers{end+1} = struct('type', 'bfc', 'weight', Winit{3}) ;
% net.layers{end+1} = struct('type', 'shrinkage','epsilon', 0.05) ;%0.005
% net.layers{end+1} = struct('type', 'bfc', 'weight', Winit{4}) ;%Emb
% net.layers{end+1} = struct('type', 'shrinkage','epsilon', 0.01) ;%0.005
% net.layers{end+1} = struct('type', 'bfc', 'weight', Winit{5}) ;%Emb
net.layers{end+1} = struct('type', 'siamese_loss', 'margin', 5) ;

%% delete net-epoch
modelPath = @(ep) fullfile(opts.dataDir, sprintf('net-epoch-%d.mat', ep)) ;
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
opts.mdmLabels = {'MDM'};
opts.train = find(covD_Struct.set==1) ;
opts.val = find(covD_Struct.set==2) ;
trn_y = covD_Struct.label(opts.train);
tst_y = covD_Struct.label(opts.val);

%% pair
Mean_spd = mean_covariances(covD_Struct.spd, 'riemann');
Mean_spd_12 = Mean_spd^(-0.5);
for i = 1 : length(covD_Struct.label)
    covD_Struct.spd(:,:,i) = Mean_spd_12*covD_Struct.spd(:,:,i)*Mean_spd_12;
end

G = Harandi_adjmat2(covD_Struct.spd(:,:,opts.train), trn_y, 20) ;%2
[train(1,:),train(2,:)] = find(G~=0);
train = train(:, randperm(length(train))) ;

% q = RandStream('mt19937ar', 'Seed', 1) ;
% val = [opts.val ; randsample(q, opts.val, numel(opts.val))] ;

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
        if epoch > 1
            fprintf('resuming by loading epoch %d\n', epoch-1) ;
            load(modelPath(epoch-1), 'net', 'info') ;
        end
    end
    
    
    %% process_epoch
    [net, stats.train] = process_epoch(opts, epoch, covD_Struct.spd, covD_Struct.label, train, learningRate, net) ;
    
    %% error
    emb = get_emb( covD_Struct.spd, net, 'net') ;
    result_ = ['epoch = ', num2str(epoch)];
    num_emb = length(fieldnames(emb));
    for layer = 1 : num_emb
        eval(['emb_ = emb.emb',num2str(layer),' ;']);
        emb_trnX = emb_(:,:,opts.train);  %#ok<IDISVAR>
        emb_tstX = emb_(:,:,opts.val);
        
        Ytest = mdm(emb_tstX, emb_trnX, trn_y);
        acc_ = 100*mean(Ytest == tst_y);
        eval(['info.train.mdm',num2str(layer),'(epoch) = acc_;']);
        result_ = strcat(result_, [', layer ',  num2str(layer), ': ' , num2str(acc_)]);
    end
    info.train.result{epoch} = result_;
    disp(result_);
    
    %% save
    evaluateMode = 0 ;
    if evaluateMode
        sets = {'train', 'val'} ;
    else
        sets = {'train'} ;
    end
    for f = sets
        f = char(f) ; %#ok<FXSET>
        n = numel(eval(f)) ;
        info.(f).objective(epoch) = stats.(f)(2) / n ;
    end
    if ~evaluateMode
        save(modelPath(epoch), 'net', 'info') ;
    end
    figure(1) ; clf ;
    hasMDM = 1 ;
    subplot(1, 1+hasMDM, 1) ;
    semilogy(1:epoch, info.train.objective, '.-', 'linewidth', 2) ;
    xlabel('training epoch') ; ylabel('energy') ;
    grid on ;
    h=legend(sets) ;
    set(h, 'color', 'none') ;
    title('convergence curve') ;
    
    if hasMDM
        subplot(1, 1+hasMDM, 2) ;
        hold on ; grid on ; legacc = {};
        for layer = 1 : num_emb
            plot(1:epoch, eval(['info.train.mdm', num2str(layer)])', '.--') ;
            legacc = horzcat(legacc, strcat('layer- ', num2str(layer))) ;
        end
        set(legend(legacc{:}), 'color', 'none') ;
        xlabel('training epoch') ; ylabel('accuracy') ;
        title('MDM curve') ;
        
        suptitle_ = suptitle([title_, ', epoch : ', num2str(epoch),...
            ',  mdm = ', num2str(eval(['info.train.mdm',num2str(num_emb),'(epoch)']))]);
    end
    set(suptitle_,'FontSize',12);
    drawnow ;
    print(1, modelFigPath, '-dpdf') ;
    epochTime = toc(Time) ;
    fprintf('On Train Set : siamese loss at epoch %02d : %d\n', epoch, info.train.objective(epoch)) ;
    %     fprintf('On Test  Set : accuracy at epoch %02d : %d\n', epoch, info.train.mdm3(epoch)) ;
    fprintf('All Time at epoch %2d : %04d\n', epoch, epochTime) ;
end


%% mail
if opts.mail
    content = ' On Siamese-Net ';
    %     content = strcat(content,...
    %         eval(['info.train.mdm',num2str(num_emb),'(epoch)']));
    DataPath = modelFigPath;
    to = 'pengzhen@buaa.edu.cn';
    mailme(content, DataPath, to);
end

end
%% process_epoch
function [net, stats] = process_epoch(opts, epoch, spd_data, spd_label, trainInd, learningRate, net) %#ok<INUSL>
%     [net, stats.train] = process_epoch(opts, epoch, covD_Struct.spd, covD_Struct.label, train, learningRate, net) ;
%     [net, stats.val] = process_epoch(opts, epoch, covD_Struct.spd, covD_Struct.label, val, 0, net) ;
training = learningRate > 0 ;
if training
    mode = 'training' ;
else
    mode = 'validation' ;
end
stats = [0 ; 0] ;
numGpus = numel(opts.gpus) ;
if numGpus >= 1
    one = gpuArray(single(1)) ;
else
    one = single(1) ;
end
batchSize = opts.batchSize ;
numDone = 0 ;
for ib = 1 : batchSize : length(trainInd)
    batchTime = tic ;
    res = [] ;
    if (ib+batchSize> length(trainInd))
        batchSize_r = length(trainInd)-ib+1 ;
    else
        batchSize_r = batchSize ;
    end
    spd.data = cell(batchSize_r, 2) ;
    spd.label = zeros(batchSize_r, 1) ;
    for ib_r = 1 : batchSize_r
        spd.data{ib_r, 1} = spd_data(:,:,trainInd(1, ib+ib_r-1)) ;
        spd.data{ib_r, 2} = spd_data(:,:,trainInd(2, ib+ib_r-1)) ;
        spd.label(ib_r) =  spd_label(trainInd(1, ib+ib_r-1)) == spd_label(trainInd(2, ib+ib_r-1)) ;
    end
    net.layers{end}.label = spd.label ;
    if training
        dzdy = one ;
    else
        dzdy = [] ;
    end
    [res, net] = vl_forbackward(net, spd.data, dzdy, res) ;
    %     if strcmp(mode, 'training') %mode == 'training'
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
    %     end
    
    numDone = numDone + batchSize_r ;
    batchTime = toc(batchTime) ;
    stats = stats+[batchTime ; res(end).x] ;
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
            if strcmp(net.layers{l}.type, 'bfc')==1
                
                W=net.layers{l}.weight ;
                problemW.M = stiefelfactory(size(W, 1), size(W, 2)) ;
                W1grad = (1/batchSize)*res(l).dzdw.l ;
                W1Rgrad = (problemW.M.egrad2rgrad(W, W1grad)) ;
                W2grad = (1/batchSize)*res(l).dzdw.r ;
                W2Rgrad = (problemW.M.egrad2rgrad(W, W2grad)) ;
                net.layers{l}.weight = (problemW.M.retr(W, -thisLR*(W1Rgrad+W2Rgrad))) ; %%!!!NOTE
                
            else
                net.layers{l}.weight = net.layers{l}.weight - thisLR * (1/batchSize)* res(l).dzdw ;
            end
        end
    end
end

end
