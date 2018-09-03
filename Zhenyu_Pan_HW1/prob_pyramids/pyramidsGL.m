function [G, L] = pyramidsGL(im, N)
% [G, L] = pyramidsGL(im, N)
% Creates Gaussian (G) and Laplacian (L) pyramids of level N from image im.
% G and L are cell where G{i}, L{i} stores the i-th level of Gaussian and Laplacian pyramid, respectively. 

    G={};
    L={};
    cutoff_frequency = 3;
    filter = fspecial('Gaussian', cutoff_frequency*4+1, cutoff_frequency);
   
    size(im);  
    im_new=imfilter(im,filter);
    orig_filt_img= im_new;
    size(im_new);
    
    for i=1:5
        G{i}= imresize(im_new,.5);
        im_new=imfilter(G{i},filter);
    end
    
    size(G{1});
    a=size(imresize(G{1},2));
    b=size(orig_filt_img);
    
    if a(1)~=b(1)&& a(2)==b(2)
        orig_filt_img=[orig_filt_img ;zeros(1,a(2))]; 
    elseif a(2)~=b(2)&& a(1)==b(1)
        orig_filt_img=[orig_filt_img  zeros(a(1),1)];   
    elseif a(2)~=b(2)&& a(1)~=b(1)
        orig_filt_img=padarray(orig_filt_img,[1 1],0,'post');
    end

    L{1}= (orig_filt_img - (imfilter((imresize(G{1},2)),filter)));

    for j= 2:5
        c=size(G{j-1});
        d=size(imresize(G{j},2));
        
        if c(2)~=d(2)&& c(1)==d(1)
            G{j-1}=padarray(G{j-1},[0 1],0,'post');
        elseif c(1)~=d(1)&& c(2)==d(2)
            G{j-1}=padarray(G{j-1},[1 0],0,'post');
        elseif c(2)~=d(2)&& c(1)~=d(1)
            G{j-1}=padarray(G{j-1},[1 1],0,'post');           
        end
        
        L{j}=G{j-1}-(imfilter((imresize(G{j},2)),filter));
        
    end

end
