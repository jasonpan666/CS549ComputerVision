% Zhenyu Pan HW3 CS/RBE549 Computer Vision WPI

function errors = eigenfaces(d, subsets_to_train)

[im, person, number, subset] = readFaceImages('faces');

X = [];

im_means = zeros(length(subset));
im_stds = zeros(length(subset));

for i=1:length(subset)
    im{i} = im2double(im{i});
    im_mean = mean2(im{i});
    im_std = std2(im{i});
    im{i} = (im{i} - im_mean)/im_std;
    im_means(i) = im_mean;
    im_stds(i) = im_std;
    if any(subsets_to_train == subset(i)) 
        im2 = reshape(im{i}, [2500 1]);
        X = horzcat(X, im2);
    end
end

[t, samples] = size(X);

mu = mean(X, 2);
X = bsxfun(@minus, X, mu);

[V, D] = eig(X' * X);
[D, sorted] = sort(diag(D), 'descend');
V = V(:, sorted);

V = X*V;

for i = 1:size(V,2)
    V(:,i) = V(:,i)/norm(V(:,i));
end

fig = figure('visible','off');
for i = 1:9
    subplot(3,3,i)
    t = V(:,i);
    t = reshape(t, [50 50]);
    imagesc(t);
    axis image;
    axis off;
    colormap gray;
end
saveas(fig, strcat('output/eigenfaces_best_', strcat(int2str(d), strcat('_', strcat(mat2str(subsets_to_train), '.png')))));


projections = zeros(samples, d);
subset_person_num = zeros(samples, 1);
V_sub = V(:, 1:d);

image_num = 1;
for i=1:length(subset)
    if any(subsets_to_train == subset(i))
        x_pca = V_sub' * reshape(im{i}, [2500 1]);
        projections(image_num, :) = x_pca;
        subset_person_num(image_num) = person(i);
        image_num = image_num + 1;
    end
end

accuracies = zeros(1, 5);
predictions = zeros(1, 640);
for i=1:length(subset)
    im2 = reshape(im{i}, [2500 1]);
    x_pca = V_sub' * im2;
    idx = knnsearch(projections,x_pca');
    predictions(i) = subset_person_num(idx);
    if subset_person_num(idx) == person(i)
        accuracies(subset(i)) = accuracies(subset(i)) + 1;
    end
end

accuracies(1) = accuracies(1)/(70);
accuracies(2) = accuracies(2)/(120);
accuracies(3) = accuracies(3)/(120);
accuracies(4) = accuracies(4)/(140);
accuracies(5) = accuracies(5)/(190);

errors = 1 - accuracies;

image_locations = [6, 11, 25, 33, 49];

fig = figure('visible','off');
for i = 1:5
    image_index = image_locations(i);
    original = im{image_index};
    original = original * im_stds(image_index) + im_means(image_index); 
    
    im2 = reshape(im{image_index}, [2500 1]);
    x_pca = V_sub' * im2;
    
    reconstructed_image = mu;
    for j = 1:d
        reconstructed_image = reconstructed_image + V_sub(:, j) * x_pca(j);
    end
    
    reconstructed = reshape(reconstructed_image, [50 50]);
    reconstructed = reconstructed * im_stds(image_index) + im_means(image_index);
    
    
    subplot(2,5,i);
    imagesc(original);
    axis image;
    axis off;
    colormap gray;
    
    subplot(2,5,5 + i);
    imagesc(reconstructed);
    axis image;
    axis off;
    colormap gray;
end
saveas(fig, strcat('output/eigenfaces_person1_', strcat(int2str(d), strcat('_', strcat(mat2str(subsets_to_train), '.png')))));

image_locations = [6, 82, 157, 235, 308];

fig = figure('visible','off');
for i = 1:5
    image_index = image_locations(i);
    original = im{image_index};
    original = original * im_stds(image_index) + im_means(image_index); 
    
    im2 = reshape(im{image_index}, [2500 1]);
    x_pca = V_sub' * im2;
    
    
    reconstructed_image = mu;
    for j = 1:d
        reconstructed_image = reconstructed_image + V_sub(:, j) * x_pca(j);
    end
    
    reconstructed = reshape(reconstructed_image, [50 50]);
    reconstructed = reconstructed * im_stds(image_index) + im_means(image_index);
    
    
    subplot(2,5,i);
    imagesc(original);
    axis image;
    axis off;
    colormap gray;
    
    subplot(2,5,5 + i);
    imagesc(reconstructed);
    axis image;
    axis off;
    colormap gray;
end

saveas(fig, strcat('output/eigenfaces_different_', strcat(int2str(d), strcat('_', strcat(mat2str(subsets_to_train), '.png')))));

