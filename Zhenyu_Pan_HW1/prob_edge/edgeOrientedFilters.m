function bmap = edgeOrientedFilters(im)
    
sigma=6
[mag, theta] = orientedFilterMagnitude(im);
bmap= edge(rgb2gray(im), 'canny').* mag .^0.7;

end