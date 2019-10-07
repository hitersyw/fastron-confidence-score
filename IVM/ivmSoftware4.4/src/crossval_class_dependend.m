function [itrn,itst,NoS_trn,NoS_tst]=crossval_class_dependend(label,NoCV,maxNum)
%%[itrn,itst,NoS_trn,NoS_tst]=crossval_class_dependend(label,NoCV,[maxNum])
% crossvalidation parts depending on number of samples per class
% label: [Nx1] vector of classlabel [1:C], C number of classes
% NoCC: number of cross validation parts
% maxNum: (optional) max number of samples per class to take at all
% itrn: cell(1,NoCV) indices for training
% itst: cell(1,NoCV) indices for testing
%
% 07.02.2012 
% susanne.wenzel@uni-bonn.de
% 
% changes
% added maxNumSamples 3.12.14 sw

if nargin<3
    maxNum = inf;
end

classes = unique(label);
num_classes = numel(classes);

itrn = cell(1,NoCV);
itst = cell(1,NoCV);
NoS_trn = cell(1,NoCV);
NoS_tst = cell(1,NoCV);

idx_all = 1:numel(label);

for c = 1:num_classes
   idx_c = find(label==classes(c));
   
   if ~isinf(maxNum)
       ix_shuffle = randperm(numel(idx_c));
       idx_c = idx_c(ix_shuffle(1:min(numel(idx_c),maxNum)));
   end
   
   [itrn_c,itst_c]=crossval(numel(idx_c),NoCV);
   
   itrn = cellfun(@(i,j)[i,idx_all(idx_c(j))],itrn,itrn_c,'UniformOutput',0);
   itst = cellfun(@(i,j)[i,idx_all(idx_c(j))],itst,itst_c,'UniformOutput',0);
   
   NoS_trn_c = cellfun(@(i)numel(i),itrn_c,'UniformOutput',0);
   NoS_tst_c = cellfun(@(i)numel(i),itst_c,'UniformOutput',0);
   
   NoS_trn = cellfun(@(i,j)[i,j],NoS_trn,NoS_trn_c,'UniformOutput',0);
   NoS_tst = cellfun(@(i,j)[i,j],NoS_tst,NoS_tst_c,'UniformOutput',0);
end

% shuffle again
itrn = cellfun(@(i)i(randperm(numel(i))),itrn,'UniformOutput',0');
itst = cellfun(@(i)i(randperm(numel(i))),itst,'UniformOutput',0');

return