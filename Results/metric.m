%clc;
zz = 1; num_band = 25; 

%img = img / max(img(:));
img(img <= 0) = 0; img(img >= 1) = 1;
img = double(img * 255);

for band=1:num_band
        err_Ours(zz,band)=psnr(squeeze(img(:,:,band)),squeeze(Img(:,:,band)),256);
    end
    
    for band=1:num_band
        SSIM_Ours(zz,band)=ssim(squeeze(img(:,:,band)) / max(squeeze(img(:,:,band))),squeeze(Img(:,:,band)) / max(squeeze(Img(:,:,band))));
    end
    
    for xx=1:size(img,1)
        for yy=1:size(img,2)
            tmp0=squeeze(Img(xx,yy,:))+eps; 
            tmp1=round(squeeze(img(xx,yy,:)))+eps; 
            err_SAM_our_temp(xx,yy)=real(hyperSam(tmp1,tmp0));
        end
    end
    err_SAM_Ours(zz)=mean(mean(err_SAM_our_temp));
    
mean_PSNR=[mean(mean(err_Ours))]
mean_SSIM=[mean(mean(SSIM_Ours))]
%mean_SAM=[mean((err_SAM_Ours))]
%mean_SAM=[mean((err_SAM_Ours))]

