%%
clear; close all; clc;

[path,~]=fileparts(matlab.desktop.editor.getActiveFilename);
cd(path);
confPath ;
rng('default') ;
rng(0) ;
experment = 'Toy-data' ;  %   'Toy-data', 'EEG-date', 'EEG-ALL'

%%
if strcmp(experment, 'Toy-data')
    sigmaVal_list = 0.1:0.2:0.5; %0.1:0.2:0.5;
    noiseVal_list = 0.1:0.2:0.5;
    accuracy = cell(length(sigmaVal_list),length(noiseVal_list));
    info.spd_net = cell(length(sigmaVal_list),length(noiseVal_list));
    info.spd_mani_net = cell(length(sigmaVal_list),length(noiseVal_list));
    net.spd_net = cell(length(sigmaVal_list),length(noiseVal_list));
    net.spd_mani_net = cell(length(sigmaVal_list),length(noiseVal_list));

    for i_sigma = 1:length(sigmaVal_list)
        for i_noise = 1:length(noiseVal_list)
            close all;
            sigmaVal = sigmaVal_list(i_sigma);
            noiseVal = noiseVal_list(i_noise);
            disp('====================================');
            disp(['sigmaVal = ', num2str(sigmaVal), ', noiseVal = ', num2str(noiseVal)]);
            disp('====================================');
            opt.sigmaVal =  sigmaVal;
            opt.noiseVal =  noiseVal;
            str = clock;
            opt.filetime = clock2time(str);
            opt.nClasses = 4;
            opt.nFeatures = 10;
            opt.nAugFeatures = 30;
            opt.nPoints = 1e2;
            opt.seed = 1;
            opt.save = 1;
            opt.clear = 1;
            opt.dataDir  = fullfile('./data',experment);
            if ~exist(opt.dataDir, 'dir')
                mkdir(opt.dataDir)
            end
            covD_Struct = generate_toydata(opt);

            %%
            opts.savemodel = 0;
            opts.savefig = 1;
            opts.experment = experment;
            opts.train = find(covD_Struct.set==1) ;
            opts.val = find(covD_Struct.set==2) ;
            trn_X = covD_Struct.spd(:,:,opts.train);
            tst_X = covD_Struct.spd(:,:,opts.val);
            trn_y = covD_Struct.label(opts.train);
            tst_y = covD_Struct.label(opts.val);
            opts.nn = 10;

            Ori.Ytest = mdm(tst_X, trn_X, trn_y);
            Ori.acc = 100*mean(Ori.Ytest == tst_y);
            Ori.stats = confusionmatStats(tst_y, Ori.Ytest);

            datadim = [30, 25, 20, 10];
            finnalDim = datadim(end);

            %% SPD-Net
            % parameter setting
            opts.batchSize = 20 ;
            opts.test.batchSize = 1 ;
            opts.numEpochs = 500 ;
            opts.gpus = [] ;
            opts.learningRate = 0.01*ones(1, opts.numEpochs) ;%0.05
            opts.weightDecay = 0.0005 ;
            opts.layernum = length(datadim)-1 ;
            opts.datadim = datadim ;
            opts.continue = 0 ;
            opts.clear = 1 ;
            opts.NetName = 'SPD-Net';
            disp('------------------------------------------------------------------');
            disp(opts.NetName);
            [spd_net, info_SPD_Net] = vl_spdnet( covD_Struct, opts );
            info.spd_net{i_sigma,i_noise} = info_SPD_Net;
            out_spdnet = emb_class(spd_net, covD_Struct, opts, 'net');

            %% SPD-Mani-Net
            % parameter setting
            opts.batchSize = 20 ;
            opts.test.batchSize = 1 ;
            opts.numEpochs = 100 ;
            opts.gpus = [] ;
            opts.learningRate = 0.01*ones(1, opts.numEpochs) ;%0.05
            opts.weightDecay = 0.0005 ;
            opts.layernum = length(datadim)-1 ;
            opts.datadim = datadim ;
            opts.continue = 0 ;
            opts.clear = 1 ;
            opts.margin = 5;
            opts.NetName = 'SPD-Mani-Net';
            disp('------------------------------------------------------------------');
            disp(opts.NetName);
            [spd_mani_net, info_SPD_Mani_Net] = vl_spd_mani_net( covD_Struct, opts );
            info.spd_mani_net{i_sigma,i_noise} = info_SPD_Mani_Net;
            out_spd_mani_net = emb_class(spd_mani_net, covD_Struct, opts, 'net');

            %% Ga_DR
            disp('------------------------------------------------------------------');
            disp('Harandi : Ga_DR');
            t = tic;
            [out_Ga_DR.U, out_Ga_DR.obj, out_Ga_DR.Adj] = Harandi( trn_X, trn_y, finnalDim, opts.nn, 'riemann');
            out_Ga_DR.time = toc(t);
            out_Ga_DR = emb_class(out_Ga_DR, covD_Struct, opts, 'linear');
            %
            %% Ga_PCA
            disp('------------------------------------------------------------------');
            disp('Horev : Ga_PCA');
            t = tic;
            [out_Ga_PCA.U, out_Ga_PCA.obj, out_Ga_PCA.M] = SPD_PCA( trn_X, finnalDim, 'riemann');
            out_Ga_PCA.time = toc(t);
            out_Ga_PCA = emb_class(out_Ga_PCA, covD_Struct, opts, 'linear');

            %% DPLM
            disp('------------------------------------------------------------------');
            disp('Davoudi : DPLM');
            t = tic;
            [out_DPLM.U, out_DPLM.obj, out_DPLM.Adj, out_DPLM.M] = DPLM( trn_X, trn_y, finnalDim, opts.nn );
            % [out_DPLM.U, out_DPLM.obj, out_DPLM.M] = SPD_PCA( trn_X, finnalDim, 'riemann');

            out_DPLM.time = toc(t);
            out_DPLM = emb_class(out_DPLM, covD_Struct, opts, 'linear');

            %% RES
            accuracy{i_sigma,i_noise} = [Ori.acc, out_Ga_DR.accuracy, out_Ga_PCA.accuracy, out_DPLM.accuracy, ...
                100*(1-info_SPD_Net.val.error(end)), out_spdnet.accuracy, out_spd_mani_net.accuracy];
            saveDir = fullfile(opt.dataDir, sprintf('result-toydata-%s-%s.mat', ...
                num2str(sigmaVal), num2str(noiseVal)));
            save(saveDir, 'Ori', 'out_spdnet', 'out_spd_mani_net', 'out_Ga_DR', 'out_Ga_PCA', 'out_DPLM') ;

        end
    end
    saveDir = fullfile(opt.dataDir, 'result-toydata-all.mat');
    save(saveDir, 'info', 'accuracy') ;
    %% Show accuracy
    tab = reshape(accuracy',[length(sigmaVal_list)*length(noiseVal_list),1]);
    tabsigmaVal = repelem(sigmaVal_list,length(noiseVal_list));
    tabnoiseVal = repmat(noiseVal_list,1,length(sigmaVal_list));
    T = array2table([tabsigmaVal', tabnoiseVal', cell2mat(tab)],'VariableNames',...
        {'sigmaVal','noiseVal','Ori','Ga-DR','Ga-PCA','DPLM','SPD-Net(val)','SPD-Net','SPD-Mani-Net'});
    disp(T);
    saveDir = fullfile(opt.dataDir, 'result-toydata-table.mat');
    save(saveDir, 'T')
end
