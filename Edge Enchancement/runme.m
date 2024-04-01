clc;
clear all; close all; 
%reading the image
img = imread('airplane_grayscale.png');
img = im2double(img);

%Threshold parameters
T = 25/255; 
Lg = 255/255;
Lb = 0/255; 

%Displaying the original image
figure
imshow(img);
title('The Original Image')

%defining the masks
Robert_mask1 = [1 0; 0 -1];
Robert_mask2 = [0 1; -1 0];

%Applying the masks to the image (H1 and H2)
H1 = convolution2D(img,Robert_mask1);
H2 = convolution2D(img,Robert_mask2);

%calculating the gradient image 1
%calculating the sum of squares of H1 and H2
sum = H1.^2+H2.^2;
G = sqrt(sum(1:end-1,1:end-1));%gradient image
G = G/max(max(G));
figure
imshow(G)
title('Gradient Image 1')

%calculating the gradient image 2 

%emphasize on edges without destroying the smooth background

%if value greater than threshold, gradient value will be used
%else the original pixel value will remain 
G1 = zeros(size(G)); 
G1(FindElement(G>=T)) = G(FindElement(G>=T)); 
G1(FindElement(G<T)) = img(FindElement(G<T));

figure
imshow(G1)
title('Gradient Image 2')

%calculating the gradient image 3

%edges are set to a specific threshold (Lg)
%if value greater than threshold, threshold (Lg) value will be used
%else the original pixel value will remain 
G2 = zeros(size(G)); 
G2(FindElement(G>=T)) = Lg;
G2(FindElement(G<T)) = img(FindElement(G<T));

figure
imshow(G2)
title('Gradient Image 3')

%calculating the gradient image 4

%background is set to a specific threshold (Lb)
%if value greater than threshold, the original pixel value will remain
%else the threshold (Lb) value will be used
G3 = zeros(size(G)); 
G3(FindElement(G>=T)) = img(FindElement(G>=T));
G3(FindElement(G<T)) = Lb;

figure
imshow(G3)
title('Gradient Image 4')

%calculating the gradient image 5

%will get a binary gradient image
%if value greater than threshold, threshold (Lg) value will be used
%else the threshold (Lb) value will be used

G4 = zeros(size(G)); 
G4(FindElement(G>=T)) = Lg;
G4(FindElement(G<T)) = Lb;

figure
imshow(G4)
title('Gradient Image 5')