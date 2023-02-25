%%
clear; close all; clc;

[path,~]=fileparts(matlab.desktop.editor.getActiveFilename);
cd(path);
confPath ;
% rng('default') ;
% rng(0) ;
experment = 'Toy-data' ;  %   'Toy-data', 'EEG-date', 'EEG-ALL'

%%
if strcmp(experment, 'Toy-data')
    sigmaVal_list = 0.5; %0.1:0.2:0.5;
    noiseVal_list = 0.5;
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
            %% Combination D: SPD-Mani-Net

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
            opts.NetName = 'spd-mani-net';
            disp('------------------------------------------------------------------');
            disp(opts.NetName);
            [netD, infoD] = vl_spd_mani_net( covD_Struct, opts );
            info.D{i_sigma,i_noise} = infoD;
            out_D = emb_class(netD, covD_Struct, opts, 'net');

            %% figure
            close all;
            savefig = figure ;
            FontSize = 12;
            set(gcf,'unit','centimeters','position',[10 5 14 8.6].*[1 1 1 1]);
            set(gcf, 'DefaultAxesFontSize', FontSize);
            out = tight_subplot(1, 1, [.05 .1], [.15 .1], [.1 .05]);
            plot(cumsum(infoD.time),infoD.val.error,'-','color',"#7E2F8E",'LineWidth',1.2); hold on;
%             lag = {'Combination-A', 'Combination-B', 'Combination-C', 'Combination-D'};
            xlabel('times [s]') ; ylabel('error on test set') ; grid on ; legend(lag);
            ax = gca;
            ax.FontName = 'Times New Roman';
            Problem_Tex = sprintf('Toy-data ($\\sigma=%s,\\delta=%s$)',...
                num2str(sigmaVal),num2str(noiseVal));
            title(Problem_Tex,'Interpreter','latex');
            Problem = sprintf('toydata-%s-%s', num2str(sigmaVal), num2str(noiseVal));
            modelFigPath = fullfile('./data/Toy-data/figure', sprintf('Ablation-%s', Problem)) ;
            saveas(savefig, sprintf('%s.%s', modelFigPath,'eps'), 'epsc');
            print(savefig,'-dpng',sprintf('%s.%s', modelFigPath,'png'));
        end
    end
end