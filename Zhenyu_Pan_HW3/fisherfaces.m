% Zhenyu Pan HW3 CS/RBE549 Computer Vision WPI

function errors = fisherfaces(c, subsets_to_train)

[im, person, number, subset] = readFaceImages('faces');

X = [];
X_orig = [];

for i=1:length(subset)
    im{i} = im2double(im{i});
    im_mean = mean2(im{i});
    im_std = std2(im{i});
    im{i} = (im{i} - im_mean)/im_std;
    if any(subsets_to_train == subset(i)) 
        im2 = reshape(im{i}, [2500 1]);
        X = horzcat(X, im2);
        X_orig = horzcat(X_orig, im2);
    end
end

[t, samples] = size(X);

mu = mean(X, 2);
X = bsxfun(@minus, X, mu);

[V, D] = eig(X' * X);
[D, sorted] = sort(diag(D), 'descend');
V = V(:, sorted);

V = X*V;

d = samples - c;

W_pca = V(:, 1:d);
pca_projections = zeros(d, samples);

for i = 1:samples
    temp = W_pca' * X_orig(:, i);
    pca_projections(:, i) = temp;
end

projected_mean = mean(pca_projections, 2);

means = zeros(d, 10);
number_per_class = zeros(10, 1);
Si = zeros(d, d, 10);
Sw = zeros(d, d);
Sb = zeros(d, d);

for i=1:length(subset)
    if any(subsets_to_train == subset(i))
        index = person(i);
        x_pca = W_pca' * reshape(im{i}, [2500 1]);
        means(:, index) = means(:, index) + x_pca;
        number_per_class(index) = number_per_class(index) + 1;
    end
end


for i=1:10
    means(:, i) = means(:, i)/number_per_class(i);
end

for i=1:length(subset)
    if any(subsets_to_train == subset(i))
        index = person(i);
        x_pca = W_pca' * reshape(im{i}, [2500 1]);
        res = x_pca - means(:, index);
        res1 = res*res';
        Si(:, :, index) = Si(:, :, index) + res1;
    end
end

for i=1:10
    temp = means(:, i) - projected_mean;
    Sb = Sb + number_per_class(i) * (temp * temp');
    Sw = Sw + Si(:, :, i);
end

[V, D] = eig(Sb, Sw);
[D, sorted] = sort(diag(D), 'descend');
W_fld = V(:, sorted);

W_opt = W_pca * W_fld;

for i = 1:samples-c
    W_opt(:,i) = W_opt(:,i)/norm(W_opt(:,i));
end

W_opt = W_opt(:, 1:c-1);

subset_person_num = zeros(samples, 1);
image_num = 1;
projections = zeros(samples, c-1);

for i=1:length(subset)
    if any(subsets_to_train == subset(i))
        x_opt = W_opt' * reshape(im{i}, [2500 1]);
        projections(image_num, :) = x_opt;
        subset_person_num(image_num) = person(i);
        image_num = image_num + 1;
    end
end

accuracies = zeros(1, 5);
predictions = zeros(1, 640);
for i=1:length(subset)
    x_opt = W_opt' * reshape(im{i}, [2500 1]);
    idx = knnsearch(projections,x_opt');
    predictions(i) = subset_person_num(idx(1));
    if subset_person_num(idx) == person(i)
        accuracies(subset(i)) = accuracies(subset(i)) + 1;
    else
        t = person(i);
        u = subset_person_num(idx);
    end
end

confusionmat(person, predictions);

accuracies(1) = accuracies(1)/(70);
accuracies(2) = accuracies(2)/(120);
accuracies(3) = accuracies(3)/(120);
accuracies(4) = accuracies(4)/(140);
accuracies(5) = accuracies(5)/(190);

errors = 1 - accuracies;