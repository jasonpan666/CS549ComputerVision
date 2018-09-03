% Starter code prepared by James Hays for Computer Vision

%This feature representation is described in the handout, lecture
%materials, and Szeliski chapter 14.

function image_feats = get_bags_of_words(image_paths)
% image_paths is an N x 1 cell array of strings where each string is an
% image path on the file system.

% This function assumes that 'vocab.mat' exists and contains an N x feature vector length
% matrix 'vocab' where each row is a kmeans centroid or visual word. This
% matrix is saved to disk rather than passed in a parameter to avoid
% recomputing the vocabulary every run.

% image_feats is an N x d matrix, where d is the dimensionality of the
% feature representation. In this case, d will equal the number of clusters
% or equivalently the number of entries in each image's histogram
% ('vocab_size') below.

% You will want to construct feature descriptors here in the same way you
% did in build_vocabulary.m (except for possibly changing the sampling
% rate) and then assign each local feature to its nearest cluster center
% and build a histogram indicating how many times each cluster was used.
% Don't forget to normalize the histogram, or else a larger image with more
% feature descriptors will look very different from a smaller version of the same
% image.

% load('vocab.mat')
% vocab_size = size(vocab, 1);

N = size(image_paths);
load('vocab.mat');
vocab_size = size(vocab, 2);
image_feats = zeros(N(1), vocab_size);
step_size = 5;
bin_size = 8;

for i = 1 : size(image_paths)
    img = imread(char(image_paths(i)));
    [~, SIFT_features] = vl_dsift(single(img), 'fast', 'step', step_size, 'size', bin_size);
    D = vl_alldist2(SIFT_features, uint8(vocab));
    [~, I] = min(D,[],2);
    [freq, word] = hist(I, unique(I));
    image_feats(i, word) = freq/norm(freq);
end

fprintf('bag_sift step: %d\n', step_size);
fprintf('bag_sift bin: %d\n', bin_size);
fprintf('vocab size:%d\n', vocab_size);

end



