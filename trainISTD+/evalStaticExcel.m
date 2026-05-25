%% compute MAE in Lab color space
clear;close all;clc

maskdir = '';  
shadowdir = "D:\welemon2077\my_cv_storyCVPR\_2_SPL_PhasorFormer-Neurocomputing\PhasorFormer_result\PhasorFormer\SRD-36.51";
freedir = 'D:/welemon2077/my_cv_storyCVPR/_0_deep learning/dataset/SRD/test/non_shadow';
output_file = 'D:\welemon2077\my_cv_storyCVPR\_2_SPL_PhasorFormer-Neurocomputing\Neurocomputing\SRD_ours.xlsx';

useMask = ~isempty(maskdir);

if useMask

    maskFiles = dir(fullfile(maskdir, '*.png'));
    maskFiles = [maskFiles; dir(fullfile(maskdir, '*.jpg')); dir(fullfile(maskdir, '*.jpeg'))];
    maskNames = arrayfun(@(x) getBaseName(x.name), maskFiles, 'UniformOutput', false);
    maskMap = containers.Map(maskNames, arrayfun(@(x) fullfile(maskdir, x.name), maskFiles, 'UniformOutput', false));
end


shadowFiles = dir(fullfile(shadowdir, '*.png'));
shadowFiles = [shadowFiles; dir(fullfile(shadowdir, '*.jpg')); dir(fullfile(shadowdir, '*.jpeg'))];

freeFiles = dir(fullfile(freedir, '*.png'));
freeFiles = [freeFiles; dir(fullfile(freedir, '*.jpg')); dir(fullfile(freedir, '*.jpeg'))];


getBaseName = @(filename) lower(erase(filename, {'.png','.jpg','.jpeg'}));
shadowNames = arrayfun(@(x) getBaseName(x.name), shadowFiles, 'UniformOutput', false);
freeNames = arrayfun(@(x) getBaseName(x.name), freeFiles, 'UniformOutput', false);


shadowMap = containers.Map(shadowNames, arrayfun(@(x) fullfile(shadowdir, x.name), shadowFiles, 'UniformOutput', false));
freeMap = containers.Map(freeNames, arrayfun(@(x) fullfile(freedir, x.name), freeFiles, 'UniformOutput', false));


if useMask
    common_names = intersect(intersect(keys(maskMap), keys(shadowMap)), keys(freeMap));
else
    common_names = intersect(keys(shadowMap), keys(freeMap));
end
N = length(common_names);

if N == 0
    error('check path');
end


full_psnr = zeros(N, 1);
full_ssim = zeros(N, 1);
full_mae = zeros(N, 1);  

if useMask
    shadow_psnr = zeros(N, 1);
    shadow_ssim = zeros(N, 1);
    shadow_mae = zeros(N, 1);  
    nonshadow_psnr = zeros(N, 1);
    nonshadow_ssim = zeros(N, 1);
    nonshadow_mae = zeros(N, 1);  
    
    total_dist_shadow = 0;
    total_pixels_shadow = 0;
    total_dist_nonshadow = 0;
    total_pixels_nonshadow = 0;
end

cform = makecform('srgb2lab');

% ========== for ==========
for i = 1:N
    base = common_names{i};
    sname = shadowMap(base);
    fname = freeMap(base);
    
    if useMask
        mname = maskMap(base);
        m = imread(mname);
    end
    
    s = imread(sname);
    f = imread(fname);
    
    f = double(f) / 255;
    s = double(s) / 255;
    
    s = imresize(s, [256 256]);
    f = imresize(f, [256 256]);
    
    if useMask
        m = double(m) / 255;
        m = imresize(m, [256 256]);
        m = imbinarize(m);
        nmask = ~m;       
        smask = m;        
    else
        smask = true(size(s,1), size(s,2));
        nmask = ~smask;
    end
    
    full_psnr(i) = psnr(s, f);
    full_ssim(i) = ssim(s, f);
    
    f_lab = applycform(f, cform);
    s_lab = applycform(s, cform);
    dist = abs(f_lab - s_lab);
    full_mae(i) = sum(dist(:)) / (size(f,1) * size(f,2));
    
    if useMask

        s_shadow = s .* repmat(smask, [1 1 3]);
        f_shadow = f .* repmat(smask, [1 1 3]);
        shadow_psnr(i) = psnr(s_shadow, f_shadow);
        shadow_ssim(i) = ssim(s_shadow, f_shadow);
        

        sdist = dist .* repmat(smask, [1 1 3]);
        sumsdist = sum(sdist(:));
        sumsmask = sum(smask(:));
        shadow_mae(i) = sumsdist / sumsmask;
        
 
        total_dist_shadow = total_dist_shadow + sumsdist;
        total_pixels_shadow = total_pixels_shadow + sumsmask;
        

        s_nonshadow = s .* repmat(nmask, [1 1 3]);
        f_nonshadow = f .* repmat(nmask, [1 1 3]);
        nonshadow_psnr(i) = psnr(s_nonshadow, f_nonshadow);
        nonshadow_ssim(i) = ssim(s_nonshadow, f_nonshadow);
        

        ndist = dist .* repmat(nmask, [1 1 3]);
        sumndist = sum(ndist(:));
        sumnmask = sum(nmask(:));
        nonshadow_mae(i) = sumndist / sumnmask;
        
        total_dist_nonshadow = total_dist_nonshadow + sumndist;
        total_pixels_nonshadow = total_pixels_nonshadow + sumnmask;
    end
    
    disp(['Processed: ', base, ' (', num2str(i), '/', num2str(N), ')']);
end


if useMask
    result_table = table(common_names', full_psnr, full_ssim, full_mae, ...
        shadow_psnr, shadow_ssim, shadow_mae, ...
        nonshadow_psnr, nonshadow_ssim, nonshadow_mae, ...
        'VariableNames', {'filename', 'full_psnr', 'full_ssim', 'full_mae', ...
        'shadow_psnr', 'shadow_ssim', 'shadow_mae', ...
        'nonshadow_psnr', 'nonshadow_ssim', 'nonshadow_mae'});
else
    result_table = table(common_names', full_psnr, full_ssim, full_mae, ...
        'VariableNames', {'filename', 'full_psnr', 'full_ssim', 'full_mae'});
end


writetable(result_table, output_file, 'Sheet', 'Per Image');


fprintf('\n========== static ==========\n');
fprintf('\nFull Region:\n');
fprintf('  PSNR: %.4f ± %.4f\n', mean(full_psnr), std(full_psnr));
fprintf('  SSIM: %.4f ± %.4f\n', mean(full_ssim), std(full_ssim));
fprintf('  MAE (Lab): %.6f ± %.6f\n', mean(full_mae), std(full_mae));

if useMask
    fprintf('\nShadow Region:\n');
    fprintf('  PSNR: %.4f ± %.4f\n', mean(shadow_psnr), std(shadow_psnr));
    fprintf('  SSIM: %.4f ± %.4f\n', mean(shadow_ssim), std(shadow_ssim));
    fprintf('  MAE (Lab) - Per Image Avg: %.6f ± %.6f\n', mean(shadow_mae), std(shadow_mae));
    fprintf('  MAE (Lab) - Overall: %.6f\n', total_dist_shadow / total_pixels_shadow);
    
    fprintf('\nNon-Shadow Region:\n');
    fprintf('  PSNR: %.4f ± %.4f\n', mean(nonshadow_psnr), std(nonshadow_psnr));
    fprintf('  SSIM: %.4f ± %.4f\n', mean(nonshadow_ssim), std(nonshadow_ssim));
    fprintf('  MAE (Lab) - Per Image Avg: %.6f ± %.6f\n', mean(nonshadow_mae), std(nonshadow_mae));
    fprintf('  MAE (Lab) - Overall: %.6f\n', total_dist_nonshadow / total_pixels_nonshadow);
    
    fprintf('\nComparison (PI-Lab vs PP-Lab):\n');
    fprintf('  Shadow: PI-Lab = %.6f, PP-Lab = %.6f\n', mean(shadow_mae), total_dist_shadow / total_pixels_shadow);
    fprintf('  Non-Shadow: PI-Lab = %.6f, PP-Lab = %.6f\n', mean(nonshadow_mae), total_dist_nonshadow / total_pixels_nonshadow);
end

fprintf('\n save: %s\n', output_file);