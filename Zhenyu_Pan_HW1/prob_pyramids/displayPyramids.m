function displayPyramids(G, L)
% Displays intensity and fft images of pyramids

    figure;
    ha = tight_subplot(2,5,[.01 .03],[.1 .01],[.01 .01]);
    for i=1:5
        subplot(ha(i)),imshow(G{i});
    end
 
    for j=1:5
        subplot(ha(j+5)),imshow((L{j})+0.5);
    end
    
end