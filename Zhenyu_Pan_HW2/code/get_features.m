% Zhenyu Pan   CS/RBE549 Computer Vision    WPI

% Local Feature Stencil Code
% Written by James Hays for CS 4476/6476 @ Georgia Tech

% Returns a set of feature descriptors for a given set of interest points. 

% 'image' can be grayscale or color, your choice.
% 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
%   The local features should be centered at x and y.
% 'feature_width', in pixels, is the local feature width. You can assume
%   that feature_width will be a multiple of 4 (i.e. every cell of your
%   local SIFT-like feature will have an integer width and height).
% If you want to detect and describe features at multiple scales or
% particular orientations you can add input arguments.

% 'features' is the array of computed features. It should have the
%   following size: [length(x) x feature dimensionality] (e.g. 128 for
%   standard SIFT)

function [features] = get_features(image, x, y, feature_width)

% To start with, you might want to simply use normalized patches as your
% local feature. This is very simple to code and works OK. However, to get
% full credit you will need to implement the more effective SIFT descriptor
% (See Szeliski 4.1.2 or the original publications at
% http://www.cs.ubc.ca/~lowe/keypoints/)

% Your implementation does not need to exactly match the SIFT reference.
% Here are the key properties your (baseline) descriptor should have:
%  (1) a 4x4 grid of cells, each feature_width/4. 'cell' in this context
%    nothing to do with the Matlab data structue of cell(). It is simply
%    the terminology used in the feature literature to describe the spatial
%    bins where gradient distributions will be described.
%  (2) each cell should have a histogram of the local distribution of
%    gradients in 8 orientations. Appending these histograms together will
%    give you 4x4 x 8 = 128 dimensions.
%  (3) Each feature should be normalized to unit length
%
% You do not need to perform the interpolation in which each gradient
% measurement contributes to multiple orientation bins in multiple cells
% As described in Szeliski, a single gradient measurement creates a
% weighted contribution to the 4 nearest cells and the 2 nearest
% orientation bins within each cell, for 8 total contributions. This type
% of interpolation probably will help, though.

% You do not have to explicitly compute the gradient orientation at each
% pixel (although you are free to do so). You can instead filter with
% oriented filters (e.g. a filter that responds to edges with a specific
% orientation). All of your SIFT-like feature can be constructed entirely
% from filtering fairly quickly in this way.

% You do not need to do the normalize -> threshold -> normalize again
% operation as detailed in Szeliski and the SIFT paper. It can help, though.

% Another simple trick which can help is to raise each element of the final
% feature vector to some power that is less than one.

%Placeholder that you can delete. Empty features.
features = zeros(size(x,1), 128);
height = size(image,1);
width = size(image,2);
filter = fspecial('gaussian',[3 3],2);
filtered_image = imfilter(image,filter);
[Gmag,Gdir] = imgradient(filtered_image);
for i = 1:size(x,1)
    current_x = x(i,:);
    current_y = y(i,:);
    if current_x > feature_width/2 & current_x < (width - feature_width/2) & current_y > feature_width/2 & current_y < (height - feature_width/2)
        Gmag_patch = Gmag(current_y-(feature_width/2)+1:current_y+feature_width/2,current_x-(feature_width/2)+1:current_x+feature_width/2);
        Gdir_patch = Gdir(current_y-(feature_width/2)+1:current_y+feature_width/2,current_x-(feature_width/2)+1:current_x+feature_width/2);
        feature = [];
        for j = 1 : 4 : size(Gmag_patch,1)
            for k = 1 : 4 : size(Gmag_patch,2)
                mag_cell = Gmag_patch(j : j+3,k : k+3);
                dir_cell = Gdir_patch(j : j+3, k : k+3);
                bin = zeros(1,8);
                for m =1:size(mag_cell,1)
                    for n=1:size(mag_cell,2)
                        if dir_cell(m,n) >= 45 && dir_cell(m,n) < 90
                            bin(1) = bin(1) + mag_cell(m,n);
                        elseif dir_cell(m,n) >= 100 && dir_cell(m,n) < 135
                            bin(2) = bin(2) + mag_cell(m,n);
                        elseif dir_cell(m,n) >= 135 && dir_cell(m,n) < 180
                            bin(3) = bin(3) + mag_cell(m,n);
                        elseif dir_cell(m,n) >= 180 && dir_cell(m,n) < -135
                            bin(4) = bin(4) + mag_cell(m,n);
                        elseif dir_cell(m,n) >= -135 && dir_cell(m,n) < -90
                            bin(5) = bin(5) + mag_cell(m,n);
                        elseif dir_cell(m,n) >= -90 && dir_cell(m,n) < -45
                            bin(6) = bin(6) + mag_cell(m,n);
                        elseif dir_cell(m,n) >= -45 && dir_cell(m,n) < 0
                            bin(7) = bin(7) + mag_cell(m,n);
                        else
                            bin(8) = bin(8) + mag_cell(m,n);
                        end
                    end   
                end
                feature = [feature bin];
            end
        end
        features(i,:) = feature./norm(feature);
    end
end
features = features .^ .7;

end








