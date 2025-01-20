clc;
[H, W, num_band] = size(img);

%img = img / max(img(:));
img(img <= 0) = 0; img(img >= 1) = 1;
img = double(img * 255);

for band=1:num_band
	PSNR_Ours(band)=psnr(squeeze(img(:,:,band)),squeeze(Img(:,:,band)),256);
end
    
for band=1:num_band
    SSIM_Ours(band)=ssim(squeeze(img(:,:,band)) / max(squeeze(img(:,:,band))),squeeze(Img(:,:,band)) / max(squeeze(Img(:,:,band))));
end
    
for xx=1:size(img,1)
    for yy=1:size(img,2)
        tmp0=squeeze(Img(xx,yy,:))+eps; 
        tmp1=round(squeeze(img(xx,yy,:)))+eps; 
        SAM_our_temp(xx,yy)=real(hyperSam(tmp1,tmp0));
    end
end
SAM_Ours = mean(mean(SAM_our_temp));

Err=img-Img;
ERGAS_index=0;
for band=1:num_band
	ERGAS_index = ERGAS_index+mean2(Err(:,:,band).^2)/(mean2((Img(:,:,band))))^2;   
end
mean_ergas = (100/1) * sqrt((1/num_band) * ERGAS_index);
    
    
    
mean_PSNR=[mean(PSNR_Ours)]
std_PSNR=[std(PSNR_Ours)]
mean_SSIM=[mean(SSIM_Ours)]
std_SSIM=[std(SSIM_Ours)]
mean_SAM=[mean(SAM_Ours)]
std_SAM=[std(SAM_our_temp(:))]
mean_ergas

