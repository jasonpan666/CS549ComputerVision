% ---------------------------------
% ---------------------------------
% Function readFaceImages()
% A helper function for reading images from an image Directory
% D. Hoiem
% Useage:   [im, person, number, subset] = readFaceImages('faces')
% Input:    imdir - Data Directory (String)
% Ouput:    im - an array of images in {cell} format
%           number - lighting condition index [1,64]
%           person - person index [1,10]
%           subset - grouping based on lighting condition [1,5]
% ---------------------------------

% Zhenyu Pan HW3 CS/RBE549 Computer Vision WPI

function [im, person, number, subset] = readFaceImages(imdir)

files = dir(fullfile(imdir, '*.png'));
for f = 1:numel(files)
  fn = files(f).name;
  person(f) = str2num(fn(7:8));
  number(f) = str2num(fn(10:11));
  if number(f) <= 7
    subset(f) = 1;
  elseif number(f) <= 19
    subset(f) = 2;
  elseif number(f) <= 31
    subset(f) = 3;
  elseif number(f) <= 45
    subset(f) = 4;
  elseif number(f) <= 64
    subset(f) = 5;
  end
  im{f} = im2single(imread(fullfile(imdir, fn)));
end
