clc;clear all;close all;

img  = imread('lena512noisy.bmp');
%img  = imread('lena512.bmp');

%%%
%decomposing the image to 16 subbands
[LoD,HiD] = wfilters('haar','d'); % 'd' decomposition filters
[LL,LH,HL,HH] = dwt2(img,LoD,HiD); %1st level decomposition

%2nd level decomposition
[i1,i2,i3,i4] = dwt2(LL,LoD,HiD);

%3rd level decomposition
[j1,j2,j3,j4] = dwt2(i1,LoD,HiD);

%4th level decomposition
[k1,k2,k3,k4] = dwt2(j1,LoD,HiD);

%5th level decomposition
[l1,l2,l3,l4] = dwt2(k1,LoD,HiD);

[LoR,HiR] = wfilters('haar','r'); % 'r' reconstruction filters

%reconstruction with one highest frequency set to 0
k1_r1 = idwt2(l1,l2,l3,l4,LoR,HiR);
j1_r1 = idwt2(k1_r1,k2,k3,k4,LoR,HiR);
i1_r1 = idwt2(j1_r1,j2,j3,j4,LoR,HiR);
LL_r1 = idwt2(i1_r1,i2,i3,i4,LoR,HiR);

img_r1 = idwt2(LL_r1,LH,HL,[],LoR,HiR);
img_r1 = uint8(img_r1);

%displaying the reconstructed image 1
figure()
imshow(img_r1)
title('Image reconstructed with one highest value set to zero')

%reconstruction with three highest frequency set to 0

img_r2 = idwt2(LL_r1,[],[],[],LoR,HiR);
img_r2 = uint8(img_r2);

%displaying the reconstructed image 2
figure()
imshow(img_r2)
title('Image reconstructed with three highest value set to zero')

%reconstruction with six highest frequency set to 0

LL_r1 = idwt2(i1_r1,[],[],[],LoR,HiR);

img_r3 = idwt2(LL_r1,[],[],[],LoR,HiR);
img_r3 = uint8(img_r3);

%displaying the reconstructed image 3
figure()
imshow(img_r3)
title('Image reconstructed with six highest value set to zero')

%plotting the DFTs of the reconstructed images\
% calculating the magnitude of the reconstructed images
img_16_dft_mag1 = abs(fftshift(fft2(img_r1))); %for reconstructed image 1
img_16_dft_mag2 = abs(fftshift(fft2(img_r2))); %for reconstructed image 2
img_16_dft_mag3 = abs(fftshift(fft2(img_r3))); %for reconstructed image 3

%plotting the DFT of the reconstructed image 1
figure
subplot(121)
imshow(img_r1)
title('Reconstructed image 1 in Spatial domain')

subplot(122)
imshow(log(1+img_16_dft_mag1), [])
title('Reconstructed image 1 in freq. domain')

%plotting the DFT of the reconstructed image 2
figure
subplot(121)
imshow(img_r2)
title('Reconstructed image 2 in Spatial domain')

subplot(122)
imshow(log(1+img_16_dft_mag2), [])
title('Reconstructed image 2 in freq. domain')

%plotting the DFT of the reconstructed image 3
figure
subplot(121)
imshow(img_r3)
title('Reconstructed image 3 in Spatial domain')

subplot(122)
imshow(log(1+img_16_dft_mag3), [])
title('Reconstructed image 3 in freq. domain')

%%%
%%%

%decomposing the image to 22 subbands
[LoD,HiD] = wfilters('haar','d'); % 'd' decomposition filters

[ll,lh,hl,hh] = dwt2(img,LoD,HiD); %1st level decomposition

%2nd level decomposition layer
[I1,I2,I3,I4] = dwt2(ll,LoD,HiD);
[I5,I6,I7,I8] = dwt2(lh,LoD,HiD);
[I9,I10,I11,I12] = dwt2(hl,LoD,HiD);
[I13,I14,I15,I16] = dwt2(hh,LoD,HiD);

%3rd level decomposition
[j1,j2,j3,j4] = dwt2(I1,LoD,HiD);

%4th level decomposition
[k1,k2,k3,k4] = dwt2(j1,LoD,HiD);


[LoR,HiR] = wfilters('haar','r'); % 'r' reconstruction filters

%reconstruction with three highest frequency set to 0
z = zeros([size(I16)]);

j1_r = idwt2(k1,k2,k3,k4,LoR,HiR);

I1_r = idwt2(j1_r,j2,j3,j4,LoR,HiR);

LL_r1 = idwt2(I1_r,I2,I3,I4,LoR,HiR);
LH_r1 = idwt2(I5,I6,I7,I8,LoR,HiR);
HL_r1 = idwt2(I9,I10,I11,I12,LoR,HiR);
HH_r1 = idwt2(I13,z,z,z,LoR,HiR);

img_r1_s22 = idwt2(LL_r1,LH_r1,HL_r1,HH_r1,LoR,HiR);
img_r1_s22 = uint8(img_r1_s22);

%displaying the reconstructed image 4
figure()
imshow(img_r1_s22)
title('Image reconstructed with three highest value set to zero')

%reconstruction with 10 highest frequency set to 0
LH_r2 = idwt2(I5,z,z,z,LoR,HiR);
HL_r2 = idwt2(I9,z,z,z,LoR,HiR);
HH_r2 = idwt2(z,z,z,z,LoR,HiR);

img_r2_s22 = idwt2(LL_r1,LH_r2,HL_r2,HH_r2,LoR,HiR);
img_r2_s22 = uint8(img_r2_s22);

%displaying the reconstructed image 5
figure()
imshow(img_r2_s22)
title('Image reconstructed with ten highest value set to zero')

%reconstruction with 15 highest frequency set to 0
LL_r3 = idwt2(I1_r,z,z,z,LoR,HiR);
LH_r3 = idwt2(I5,z,z,z,LoR,HiR);
HL_r3 = idwt2(I9,z,z,z,LoR,HiR);
HH_r3 = idwt2(z,z,z,z,LoR,HiR);

img_r3_s22 = idwt2(LL_r3,LH_r3,HL_r3,HH_r3,LoR,HiR);
img_r3_s22 = uint8(img_r3_s22);

%displaying the reconstructed image 6
figure()
imshow(img_r3_s22)
title('Image reconstructed with fifteen highest value set to zero')

%plotting the DFTs of the reconstructed images\
% calculating the magnitude of the reconstructed images
img_16_dft_mag1_s22 = abs(fftshift(fft2(img_r1_s22))); %for reconstructed image 1
img_16_dft_mag2_s22 = abs(fftshift(fft2(img_r2_s22))); %for reconstructed image 2
img_16_dft_mag3_s22 = abs(fftshift(fft2(img_r3_s22))); %for reconstructed image 3

%plotting the DFT of the reconstructed image 4
figure
subplot(121)
imshow(img_r1_s22)
title('Reconstructed image 4 in Spatial domain')

subplot(122)
imshow(log(1+img_16_dft_mag1_s22), [])
title('Reconstructed image 4 in freq. domain')

%plotting the DFT of the reconstructed image 5
figure
subplot(121)
imshow(img_r2_s22)
title('Reconstructed image 5 in Spatial domain')

subplot(122)
imshow(log(1+img_16_dft_mag2_s22), [])
title('Reconstructed image 5 in freq. domain')

%plotting the DFT of the reconstructed image 6
figure
subplot(121)
imshow(img_r3_s22)
title('Reconstructed image 6 in Spatial domain')

subplot(122)
imshow(log(1+img_16_dft_mag3_s22), [])
title('Reconstructed image 6 in freq. domain')

%%%
% plotting the lena512noisy and it's dft
figure
imshow(img)
title('lena512noisy in Spatial domain')

img_dft= abs(fftshift(fft2(img)));

figure
imshow(log(1+img_dft), [])
title('lena512noisy in freq. domain')
