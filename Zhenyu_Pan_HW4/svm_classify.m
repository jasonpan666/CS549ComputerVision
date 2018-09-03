% Starter code prepared by James Hays for Computer Vision

%This function will train a linear SVM for every category (i.e. one vs all)
%and then use the learned linear classifiers to predict the category of
%every test image. Every test feature will be evaluated with all 15 SVMs
%and the most confident SVM will "win". Confidence, or distance from the
%margin, is W*X + B where '*' is the inner product or dot product and W and
%B are the learned hyperplane parameters.

function predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)
% image_feats is an N x d matrix, where d is the dimensionality of the
%  feature representation.
% train_labels is an N x 1 cell array, where each entry is a string
%  indicating the ground truth category for each training image.
% test_image_feats is an M x d matrix, where d is the dimensionality of the
%  feature representation. You can assume M = N unless you've modified the
%  starter code.
% predicted_categories is an M x 1 cell array, where each entry is a string
%  indicating the predicted category for each test image.

%{
Useful functions:
 matching_indices = strcmp(string, cell_array_of_strings)
 
  This can tell you which indices in train_labels match a particular
  category. This is useful for creating the binary labels for each SVM
  training task.

[W B] = vl_svmtrain(features, labels, LAMBDA)
  http://www.vlfeat.org/matlab/vl_svmtrain.html

  This function trains linear svms based on training examples, binary
  labels (-1 or 1), and LAMBDA which regularizes the linear classifier
  by encouraging W to be of small magnitude. LAMBDA is a very important
  parameter! You might need to experiment with a wide range of values for
  LAMBDA, e.g. 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10.

  Matlab has a built in SVM, see 'help svmtrain', which is more general,
  but it obfuscates the learned SVM parameters in the case of the linear
  model. This makes it hard to compute "confidences" which are needed for
  one-vs-all classification.

%}

%unique() is used to get the category list from the observed training
%category list. 'categories' will not be in the same order as in proj4.m,
%because unique() sorts them. This shouldn't really matter, though.
%SVM
global K X Y

categories = unique(train_labels); 
num_categories = length(categories);
num_train_image_feats = size(train_image_feats);
num_test_image_feats = size(test_image_feats);
n = num_train_image_feats(1);
nt = num_test_image_feats(1);

% Free params for linear SVM
lambda = .000001;

% Free params and global variables for training non-linear SVM
X = train_image_feats;
hp.type = 'rbf';
hp.sig = 1;
K = compute_kernel(train_image_feats,train_image_feats, 1:n, 1:n, hp);

W = [];
B = [];
for i=1:num_categories
    label_logical = strcmp(categories(i),train_labels);
    Y = ones(size(label_logical)) .* -1;
    Y(label_logical) = 1;
    [w,b] = primal_svm(0, Y, lambda);
    %[w,b] = vl_svmtrain(train_image_feats',Y',lambda);
    W(i,:) = w;
    B(i,:) = b;
end

K_test = compute_kernel(test_image_feats,train_image_feats, 1:nt, 1:nt, hp);

for i = 1:nt
    weight = [];
    for j=1:num_categories
        %weight(j,:) = dot(W(j,:), test_image_feats(i,:)) + B(j,:); 
        weight(j,:) = dot(W(j,:), K_test(i,:)) + B(j,:); 
    end    
    [~,ind] = max(weight);
    predicted_categories(i) = categories(ind);
end
predicted_categories = predicted_categories';

end