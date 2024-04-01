clc; clear all; close all; 
%reading the lena.raw  image provided
fid = fopen('lena.raw','r');
I = fread(fid,[512 512],'uint8=>uint8');
fclose(fid);
I = I';

[rows, columns] = size(I);%getting the dimensions of the image
block_size = 8; %for dividing the image

%displaying the original iamge
figure
imshow(uint8(I));
title('Original Image');
figure
%calculating the mag spectrum of DFT of the original iamge
dft_mag_spectrum = abs(fftshift(fft2(double(I)))); 
imshow(log(1 + dft_mag_spectrum), []);
title('2D DFT Magnitude spectrum of Original Image');

%determining blocks for each dimension
num_blocks_rows = rows/ block_size;
num_blocks_columns= columns /block_size  ;

% A cell array to store the divied blocks 
blocks = mat2cell(I, repmat(block_size, 1, num_blocks_rows), repmat(block_size, 1, num_blocks_columns));
dct_blocks = cell(num_blocks_rows, num_blocks_columns);%to store the dct results of the divided blocks

for i = 1:num_blocks_rows
    for j = 1:num_blocks_columns
        block = blocks{i,j};
        dct_block = dct_2d(block);%applying dct to ecah block using the fucntion made from
         % scratch ---dct_2d
        dct_blocks{i,j} = dct_block;% storing the dct blocks in to the array cell
    end
end

coeff = [50,75,90,95];

zer = 0; 
quant_dct_blocks = cell(num_blocks_rows, num_blocks_columns);

for k = 1:4
    mask_name = sprintf('mask%d.mat', k);
    mask = struct2array(load(mask_name));% mask is to zero the coeffients
 %zeroing the coeffients   
    for i = 1:num_blocks_rows
    for j = 1:num_blocks_columns
    dct_block = dct_blocks{i,j};
    % storing the dct blocks in to the array cell after zzeroing the coefficients
    dct_blocks{i,j} = dct_block.*mask; 
    end
    end

quant_step = 2^3;

quant_dct_blocks = cell(num_blocks_rows, num_blocks_columns);
for i = 1:num_blocks_rows
    for j = 1:num_blocks_columns
        dct_block = dct_blocks{i,j};
        % Quantizing the coefficients using an 8-bit uniform scalar quantizer
        sign_dct_block = sign(dct_block);%to preserve the sign
        abs_dct_block = abs(dct_block);
        quant_abs_dct_block = round(abs_dct_block/quant_step)*quant_step;
        quant_dct_block = sign_dct_block .* quant_abs_dct_block;
        quant_dct_blocks{i,j} = quant_dct_block ;% storing the quantized dct blocks into the array cell
        
    end
end


% Reconstructing the image
reconstructed_image = zeros(rows, columns);
for i =1:num_blocks_rows
    for j= 1:num_blocks_columns
         % applying inverse 2d DCT on each quantized block using the fucntion made from
         % scratch ---invdct2
      idct_block = invdct2(quant_dct_blocks{i,j});
        reconstructed_image((i-1)*block_size+1:i*block_size,(j-1)*block_size+1:j*block_size) = idct_block;
        
    end
end


%displaying the reconstructed image
figure;
imshow(uint8(reconstructed_image));
title(sprintf('Reconstructed Image with %d%% Coefficients Zeroed Out', coeff(k)));

% Displaying the 2D DFT magnitude spectrum of the reconstructed image
figure;
dft_mag_spectrum = abs(fftshift(fft2(double(reconstructed_image))));
imshow(log(1 + dft_mag_spectrum), []);
title(sprintf('2D DFT Magnitude spectrum of Reconstructed Image with %d%% Coefficients Zeroed Out', coeff(k)));

%Calculating the PSNR
MSE = mean(mean((double(I)-reconstructed_image).^2)); % Mean Squared Error
PSNR = 10*log10((255^2)/MSE); %Peak SNR

fprintf('PSNR = %0.2f dB\n', PSNR);
   
end
