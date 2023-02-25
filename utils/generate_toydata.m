function covD_Struct = generate_toydata(option)

dataDir = fullfile(option.dataDir, sprintf('toydata-%1.0f-%1.0f.mat', ...
    10*option.sigmaVal, 10*option.noiseVal));

if option.clear && exist(dataDir,"file")
    delete(dataDir);
elseif exist(dataDir,"file")
    load(dataDir,'covD_Struct');
    return;
end
% if exist(dataDir,"file") && ~option.clear
%     load(dataDir,'covD_Struct');
%     return;
% end

rng(option.seed);
% s = RandStream('mt19937ar','seed',option.seed);
% RandStream.setDefaultStream(s);

nClasses = option.nClasses;
nFeatures = option.nFeatures;
nAugFeatures = option.nAugFeatures;
nPoints = option.nPoints;
sigmaVal =  option.sigmaVal;
noiseVal =  option.noiseVal;

% % % nTraining = round(2*nPoints/4);
nTraining = round(nPoints/2);
%Generate points on identity tangent space
covD_Struct.trn_X = [];
covD_Struct.trn_y = [];
covD_Struct.tst_X = [];
covD_Struct.tst_y = [];
tmpCNTR = 1; %#ok<NASGU>
% m = rand(nFeatures*(nFeatures+1)/2,nClasses);
m0 = repmat(rand(nFeatures*(nFeatures+1)/2,1),[1 nClasses]);
m = m0 +0.5*rand(size(m0,1),nClasses);
feature_max = zeros(nAugFeatures,1);
for tmpC1 = 1:nClasses
    tmpPoints =  repmat(m(:,tmpC1),[1,nPoints]) + sigmaVal*randn(size(m,1),nPoints);
    tmpPoints = [tmpPoints;noiseVal*randn((nAugFeatures*(nAugFeatures+1) - nFeatures*(nFeatures+1))/2,nPoints)]; %#ok<AGROW>
    tmpPoints = Euclidean2SPD(tmpPoints);
    for tmpC2 = 1:nPoints
        tmpX = expm(tmpPoints(:,:,tmpC2));
        [V,D] = eig(tmpX);
        D = diag(D + eps);
        inv_D = 1./sqrt(D+eps); %#ok<NASGU>
        %         X(:,:,tmpC2) = diag(inv_D+eps)*(V*diag(D)*V');
        X(:,:,tmpC2) = (V*diag(D)*V');  %#ok<AGROW>
        var_curr = diag(X(:,:,tmpC2));
        idx = var_curr > feature_max;
        if any(idx)
            feature_max(idx) = var_curr(idx); 
        end
    end
    covD_Struct.trn_X = cat(3,covD_Struct.trn_X,X(:,:,1:nTraining));
    covD_Struct.trn_y = [covD_Struct.trn_y tmpC1*ones(1,nTraining)];
    covD_Struct.tst_X = cat(3,covD_Struct.tst_X,X(:,:,1+nTraining:end));
    covD_Struct.tst_y = [covD_Struct.tst_y tmpC1*ones(1,nPoints - nTraining)];
end
covD_Struct.nClasses = nClasses;
covD_Struct.n = nAugFeatures;
covD_Struct.r = nFeatures;
U = diag(feature_max.^(-1/2));

%Normalizing data
for tmpC1 = 1:size(covD_Struct.trn_X,3)
    covD_Struct.trn_X(:,:,tmpC1) = U*covD_Struct.trn_X(:,:,tmpC1)*U;
end

for tmpC1 = 1:size(covD_Struct.tst_X,3)
    covD_Struct.tst_X(:,:,tmpC1) = U*covD_Struct.tst_X(:,:,tmpC1)*U;
end

covD_Struct.spd = cat(3, covD_Struct.trn_X, covD_Struct.tst_X);
covD_Struct.set = [1*ones(1, size(covD_Struct.trn_X,3)), 2*ones(1, size(covD_Struct.tst_X,3))];
covD_Struct.label = [covD_Struct.trn_y, covD_Struct.tst_y];
covD_Struct.sigmaVal = sigmaVal;
covD_Struct.noiseVal = noiseVal;
covD_Struct.time = option.filetime;
covD_Struct.Dir = option.dataDir;
covD_Struct.dataDir = fullfile(covD_Struct.Dir, sprintf('toydata-%s-%s.mat', ...
    num2str(covD_Struct.sigmaVal), num2str(covD_Struct.noiseVal)));
covD_Struct = rmfield(covD_Struct, {'trn_X','tst_X', 'trn_y', 'tst_y'});
if option.save == 1
    save(covD_Struct.dataDir, 'covD_Struct', 'option') ;
end

end