function inv_up = update_inverse(inv_old, k)

% description: update inverse if one sample is removed
% author: Ribana Roscher (rroscher@uni-bonn.de)
% date: August 2012 (last modified)

inv_up = inv_old - (inv(inv_old(k, k)) * inv_old(:, k)) * inv_old(k, :);