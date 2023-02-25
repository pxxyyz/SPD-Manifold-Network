function in = emb_class(in, covD_Struct, opts, type)
%%
opts.train = find(covD_Struct.set==1) ;
opts.val = find(covD_Struct.set==2) ;
trn_X = covD_Struct.spd(:,:,opts.train);
tst_X = covD_Struct.spd(:,:,opts.val);
trn_y = covD_Struct.label(opts.train);
tst_y = covD_Struct.label(opts.val);

%%
if strcmp(type, 'linear')
    %%
    newDim = size(in.U, 2);
    emb_trnX = zeros(newDim,newDim,length(trn_y));
    for tmpC1 = 1:length(trn_y)
        emb_trnX(:,:,tmpC1) = in.U'*trn_X(:,:,tmpC1)*in.U;
    end
    emb_tstX = zeros(newDim,newDim,length(tst_y));
    for tmpC1 = 1:length(tst_y)
        emb_tstX(:,:,tmpC1) = in.U'*tst_X(:,:,tmpC1)*in.U;
    end
elseif strcmp(type, 'net')
    %%
    emb = get_emb( covD_Struct.spd, in) ;
    emb_trnX = emb(:,:,opts.train);
    emb_tstX = emb(:,:,opts.val);
end
%%
in.Ytest = mdm(emb_tstX, emb_trnX, trn_y);
in.accuracy = 100*mean(in.Ytest == tst_y);
in.stats = confusionmatStats(tst_y, in.Ytest);
end