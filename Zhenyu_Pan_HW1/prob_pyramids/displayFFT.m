function displayFFT(G, L)   
% Displays FFT images

    figure;
    for i=1:5
        subplot(2,5,i);
        imagesc((log((abs(fftshift(fft2(double(G{i}))))))));
        colormap jet;
        subplot(2,5,i+5);
        imagesc((log((abs(fftshift(fft2(double(L{i}+0.5))))))));
        colormap jet;
    end

end
