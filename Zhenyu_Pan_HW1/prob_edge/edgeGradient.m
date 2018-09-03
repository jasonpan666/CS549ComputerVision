function bmap = edgeGradient(im)

sigma=5
[mag, theta] = gradientMagnitude(im, sigma);
bmap= edge(rgb2gray(im), 'canny').*mag.^0.7;

end