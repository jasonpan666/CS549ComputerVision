function [ mag, theta ] = orientedFilterMagnitude(im)
% Funtion which applys oriented filters to image im, and output a single 
% Gradient image 

% Step 1 : Parameters for the oriented Gaussian filter set
sigx = 10; sigy = 1; width = 4*sigx+1; offset = 0; factor = 1;

%orientations = [0:30:180];
orientations = [0:20:180];
fColumn = length(orientations);
grayim = rgb2gray(im);
%F = fspecial('laplacian', 0.2);
%im = imfilter(im,F);

% Step 2 :  Create normalized oriented Gaussian filter set --%
G_filter = cell([1, fColumn]);
for ii = 1:fColumn
    G_filter{ii} = customgauss([width,width],sigx, sigy,orientations(ii),offset,factor,[0,0]);
    add = sum(sum(G_filter{ii}));
    G_filter{ii} = G_filter{ii}/add;
end

imSize = size(grayim);
imRow = imSize(1); imColumn = imSize(2);

% Step 3 : Separate image into R G B channels 
RGB_layers = cell([1,3]);
RGB_layers{1} = im(:,:,1); RGB_layers{2} = im(:,:,2); RGB_layers{3} = im(:,:,3);

mag_array = cell([1,fColumn]);
theta_array = cell([1,fColumn]);

ColorMag_array = cell([1,3]);
ColorTheta_array = cell([1,3]);

mag = zeros(imSize);
theta = zeros(imSize);

for nn = 1:3
    % apply all filter to one color channel
    for ii = 1:fColumn
        [mag_array{ii}, theta_array{ii}] = imgradient( imfilter(RGB_layers{nn}, G_filter{ii}) );
    end 

    % find best mag and theta 
    for ii = 1:imRow
        for jj = 1:imColumn
            mag_c = [mag_array{1}(ii,jj), mag_array{2}(ii,jj), mag_array{3}(ii,jj), mag_array{4}(ii,jj), mag_array{5}(ii,jj), mag_array{6}(ii,jj), mag_array{7}(ii,jj), mag_array{8}(ii,jj), mag_array{9}(ii,jj), mag_array{10}(ii,jj)];
            theta_c = [theta_array{1}(ii,jj), theta_array{2}(ii,jj), theta_array{3}(ii,jj), theta_array{4}(ii,jj), theta_array{5}(ii,jj), theta_array{6}(ii,jj), theta_array{7}(ii,jj), theta_array{8}(ii,jj), theta_array{9}(ii,jj), theta_array{10}(ii,jj)];
            ColorMag_array{nn}(ii,jj) = max(mag_c);
            ColorTheta_array{nn}(ii,jj) = max(theta_c);
        end
    end
end

% Step 4 :  Merge three channels to form final mag and theta output
for ii = 1:imRow
    for jj = 1:imColumn
        c = [ColorMag_array{1}(ii,jj),ColorMag_array{2}(ii,jj),ColorMag_array{3}(ii,jj)];
        maxmag = max(c);
        if maxmag == c(1)
            theta(ii,jj) = ColorTheta_array{1}(ii,jj);
        elseif maxmag == c(2)
            theta(ii,jj) = ColorTheta_array{2}(ii,jj);
        else
            theta(ii,jj) = ColorTheta_array{3}(ii,jj);
        end
        mag(ii,jj) = norm(c);
    end
end

end