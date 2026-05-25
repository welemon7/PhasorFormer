%% compute RMSE(MAE)|RMSE(MAE) 
clear;close all;clc

maskdir = 'D:/welemon2077/my_cv_storyCVPR/_0_deep learning/dataset/SRD/test/mask';
shadowdir = "D:\welemon2077\my_cv_storyCVPR\_2_SPL_PhasorFormer-Neurocomputing\PhasorFormer_result\PhasorFormer\SRD-36.51";
freedir = 'D:/welemon2077/my_cv_storyCVPR/_0_deep learning/dataset/SRD/test/non_shadow';


maskFiles = dir(fullfile(maskdir, '*.png'));
maskFiles = [maskFiles; dir(fullfile(maskdir, '*.jpg')); dir(fullfile(maskdir, '*.jpeg'))];

shadowFiles = dir(fullfile(shadowdir, '*.png'));
shadowFiles = [shadowFiles; dir(fullfile(shadowdir, '*.jpg')); dir(fullfile(shadowdir, '*.jpeg'))];

freeFiles = dir(fullfile(freedir, '*.png'));
freeFiles = [freeFiles; dir(fullfile(freedir, '*.jpg')); dir(fullfile(freedir, '*.jpeg'))];

getBaseName = @(filename) lower(erase(filename, {'.png','.jpg','.jpeg'}));
maskNames = arrayfun(@(x) getBaseName(x.name), maskFiles, 'UniformOutput', false);
shadowNames = arrayfun(@(x) getBaseName(x.name), shadowFiles, 'UniformOutput', false);
freeNames = arrayfun(@(x) getBaseName(x.name), freeFiles, 'UniformOutput', false);


maskMap = containers.Map(maskNames, arrayfun(@(x) fullfile(maskdir, x.name), maskFiles, 'UniformOutput', false));
shadowMap = containers.Map(shadowNames, arrayfun(@(x) fullfile(shadowdir, x.name), shadowFiles, 'UniformOutput', false));
freeMap = containers.Map(freeNames, arrayfun(@(x) fullfile(freedir, x.name), freeFiles, 'UniformOutput', false));


common_names = intersect(intersect(keys(maskMap), keys(shadowMap)), keys(freeMap));
N = length(common_names);

if N == 0
    error('check path');
end


allmae = zeros(1, N);
smae   = zeros(1, N);
nmae   = zeros(1, N);
ppsnr  = zeros(1, N);
ppsnrs = zeros(1, N);
ppsnrn = zeros(1, N);
sssim  = zeros(1, N);
sssims = zeros(1, N);
sssimn = zeros(1, N);

total_dists  = 0;
total_pixels = 0;
total_distn  = 0;
total_pixeln = 0;

cform = makecform('srgb2lab');


for i = 1:N
    base = common_names{i};
    mname = maskMap(base);
    sname = shadowMap(base);
    fname = freeMap(base);
    
    s = imread(sname);
    f = imread(fname);
    m = imread(mname);
    

    f = double(f) / 255;
    s = double(s) / 255;
    
    s = imresize(s, [256 256]);
    f = imresize(f, [256 256]);
    m = imresize(m, [256 256]);
    
    nmask = ~m;       
    smask = ~nmask;   
    
    ppsnr(i)  = psnr(s, f);
    ppsnrs(i) = psnr(s .* repmat(smask, [1 1 3]), f .* repmat(smask, [1 1 3]));
    ppsnrn(i) = psnr(s .* repmat(nmask, [1 1 3]), f .* repmat(nmask, [1 1 3]));
    
    sssim(i)  = ssim(s, f);
    sssims(i) = ssim(s .* repmat(smask, [1 1 3]), f .* repmat(smask, [1 1 3]));
    sssimn(i) = ssim(s .* repmat(nmask, [1 1 3]), f .* repmat(nmask, [1 1 3]));
    
    f_lab = applycform(f, cform);
    s_lab = applycform(s, cform);
    
    dist = abs(f_lab - s_lab);
    sdist = dist .* repmat(smask, [1 1 3]);
    ndist = dist .* repmat(nmask, [1 1 3]);
    
    sumsdist = sum(sdist(:));
    sumndist = sum(ndist(:));
    
    sumsmask = sum(smask(:));
    sumnmask = sum(nmask(:));
    
    allmae(i) = sum(dist(:)) / size(f,1) / size(f,2);
    smae(i)   = sumsdist / sumsmask;
    nmae(i)   = sumndist / sumnmask;
    
    total_dists  = total_dists + sumsdist;
    total_pixels = total_pixels + sumsmask;
    total_distn  = total_distn + sumndist;
    total_pixeln = total_pixeln + sumnmask;
    
    disp(['Processed: ', base, ' (', num2str(i), '/', num2str(N), ')']);
end

fprintf('\n========== final ==========\n');
fprintf('Shadow region:\n');
fprintf('PSNR: %.4f\tSSIM: %.4f\tPI-Lab: %.6f\tPP-Lab: %.6f\n', ...
    mean(ppsnrs), mean(sssims), mean(smae), total_dists/total_pixels);

fprintf('Non-Shadow region:\n');
fprintf('PSNR: %.4f\tSSIM: %.4f\tPI-Lab: %.6f\tPP-Lab: %.6f\n', ...
    mean(ppsnrn), mean(sssimn), mean(nmae), total_distn/total_pixeln);

fprintf('All region:\n');
fprintf('PSNR: %.4f\tSSIM: %.4f\tPI-Lab: %.6f\tPP-Lab: %.6f\n\n', ...
    mean(ppsnr), mean(sssim), mean(allmae), mean(allmae));

fprintf('\n========== final ==========\n');
fprintf('Shadow region:\n');
fprintf('PSNR: %.4f\tSSIM: %.4f\tPI-Lab: %.6f\n', ...
    mean(ppsnrs), mean(sssims), mean(smae));

fprintf('Non-Shadow region:\n');
fprintf('PSNR: %.4f\tSSIM: %.4f\tPI-Lab: %.6f\n', ...
    mean(ppsnrn), mean(sssimn), mean(nmae));

fprintf('All region:\n');
fprintf('PSNR: %.4f\tSSIM: %.4f\tPI-Lab: %.6f\n', ...
    mean(ppsnr), mean(sssim), mean(allmae));

