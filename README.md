# Image-Processing
This repository is a collection of image processing projects I've been working on. Each project includes the source code, along with detailed explanations and steps to guide you through the process. Whether you're a beginner looking to learn the fundamentals or an experienced developer seeking new ideas, this repository can be a valuable resource for your image processing journey.

## [SAS Beamforming](https://github.com/MokshagnaRohit/Image-Processing-/blob/master/SAS%20Beamforming)
This subtopic outlines the steps involved in SAS beamforming for AirSAS measurements:

1. **Preprocessing:**
    * Apply a match filter to each AirSAS measurement.
    * Compute the analytic signal for each measurement.
    * Plot the magnitude and phase of the resulting waveforms.
2. **Delay-and-Sum Beamforming:**

   A custom function (`delay_and_sum_beamformer`) is implemented to take the following inputs:
     * AirSAS measurements
     * Pixels of interest
     * Transmitter/Receiver coordinates

   The function returns a complex image representing the beamformed data.
3. **Beamformed Image Analysis:**

    * Plot the magnitude and phase of the complex image obtained from the beamformer.
    * Compare the beamformed image with a reference optical image to assess the effectiveness of the beamforming process.

**Note:** While the beamformed image might not perfectly capture the object's exact boundaries, it should reveal its general shape.

## Image Enhancement and Filtering
### [Wiener Filtering for Image Denoising](https://github.com/MokshagnaRohit/Image-Processing-/tree/main/Wiener%20Filtering)

Wiener filtering is a technique used in signal and image processing to remove noise from a signal while preserving the original signal content. It's particularly effective for additive noise, which commonly affects images. Here's a breakdown of the steps followed to explore Wiener filtering:

**1. Image Acquisition and Blurring:**

* Load an image of a favorite animal (replace it with your choice).
* Create a custom blur kernel (e.g., averaging or Gaussian) to simulate image degradation.
* Apply the blur kernel to the original image, creating a blurred version.

**2. Naive Deconvolution:**

* Implement a naive deconvolution approach to attempt noise removal from the blurred image. This is a simple mathematical operation that can sometimes amplify noise.
* Compare the results to the original image to assess the effectiveness of deconvolution.
* Add noise (e.g., Gaussian noise) to the blurred image to simulate real-world scenarios.
* Apply naive deconvolution again, comparing the results to original and blurred images. The presence of noise will significantly affect deconvolution performance, often leading to artifacts and further degradation.

**3. Wiener Filtering vs. Naive Deconvolution:**

* Utilize a built-in library function or an existing implementation of Wiener filtering to denoise the blurred image (both with and without added noise).
* Wiener filtering takes into account the statistical properties of both the signal and noise, resulting in a more robust denoising approach compared to naive deconvolution.
* Compare the results of Wiener filtering to the blurred image, the results of naive deconvolution, and the original image. In most cases, Wiener filtering will outperform naive deconvolution, especially in the presence of noise.

**Note:** The specific implementation details (libraries, noise types, etc.) will vary depending on your chosen programming language and tools.

By comparing these techniques, we gain a better understanding of how Wiener filtering can be a valuable tool for image restoration and noise reduction.

### [Wavelet Denoising](https://github.com/MokshagnaRohit/Image-Processing-/tree/main/Wavelet%20Denoising)

This section describes the implementation of two image-denoising systems using the 2-D Stationary Wavelet Transform (SWT) in MATLAB. The goal is to explore the effectiveness of SWT for noise reduction in the "lena512noisy.bmp" image. 

**Methodology:**

1. **Image Decomposition:** The noisy image is decomposed into subbands using two different SWT schemes:
    - **16-band dyadic (pyramid) decomposition:** The image is decomposed into 16 equal-sized subbands, representing different frequency levels.
    - **22-band modified pyramid decomposition:** The image undergoes an initial decomposition into 16 subbands, followed by two additional levels of decomposition applied only to the lowest-frequency subband, resulting in a total of 22 subbands.

2. **Thresholding and Reconstruction:**
    - For the dyadic decomposition:
        - The highest-frequency subband, the three highest-frequency subbands, and the six highest-frequency subbands are assumed to be dominated by noise and set to zero.
        - The remaining coefficients (lower-frequency subbands) are left unchanged.
    - For the modified pyramid decomposition:
        - The three highest-frequency subbands, the 10 highest-frequency subbands, and the 15 highest-frequency subbands are set to zero.
        - The remaining coefficients are preserved.
    - In both cases, the inverse SWT is applied to reconstruct the image using the modified coefficients.

3. **Evaluation:**
    - The reconstructed images from each decomposition scheme are displayed.
    - The perceived quality of each reconstructed image is assessed visually, focusing on noise reduction and potential artifacts.

4. **Frequency Domain Analysis:**
    - For each case (original noisy image, reconstructed images from both decompositions), the 2-D Discrete Fourier Transform (DFT) magnitude spectrum is computed and plotted.
    - The frequency spectrum visualizations are compared, analyzing how the denoising process affects the distribution of frequency components.

**Expected Outcome:**

We anticipate that reconstructed images obtained through SWT-based thresholding will exhibit reduced noise compared to the original noisy image. The modified pyramid decomposition might offer more flexibility in noise removal due to its finer subband structure. Analyzing the visual quality and frequency spectrums will provide insights into the effectiveness of each SWT denoising approach.

**Note:** The specific SWT filter chosen in MATLAB will be mentioned in the code implementation itself (not included here). Techniques like edge padding or reflection can be used to eliminate edge effects during the SWT decomposition. 
Read more at [doc.](https://github.com/MokshagnaRohit/Image-Processing-/blob/main/Wavelet%20Denoising/Student_Report.pdf)

### [Edge enhancement](https://github.com/MokshagnaRohit/Image-Processing-/tree/main/Edge%20Enchancement)
Edge enhancement of “airplane_grayscale.png” image using Roberts gradient convolution masks to 
generate five gradient images with different values of T, LG, and LB. 
1. **Threshold (T)**: A threshold value will be used to binarize the gradient image obtained from convolution. Pixel values above the threshold will be considered edges, while those below will be set to zero.
2. **Lower bound (LB)**: This value will clip the negative values in the gradient image before thresholding.
3. **Upper bound (UB)**: This value will clip the positive values in the gradient image before thresholding.

Read more at [doc.](https://github.com/MokshagnaRohit/Image-Processing-/blob/main/Edge%20Enchancement/Documentation.pdf)

## [Image Compression using 2-D Discrete Cosine Transform (DCT)](https://github.com/MokshagnaRohit/Image-Processing-/tree/main/Image%20Compression%20Using%20DCT)

This section describes an image compression system implemented using the 2-D DCT (Two-Dimensional Discrete Cosine Transform). The goal is to explore the trade-off between compression ratio and image quality.

**Methodology:**

1. **Image Segmentation:** The original image is segmented into non-overlapping 8x8 pixel blocks.
2. **2-D DCT Transformation:** The 2-D DCT is independently applied to each block, transforming the spatial domain (pixel intensities) into the frequency domain (DCT coefficients).
3. **Coefficient Selection and Quantization:**
    - A percentage of the higher-frequency DCT coefficients within each block are set to zero, discarding information deemed less perceptually important for humans.
    - The remaining coefficients are quantized using an 8-bit uniform scalar quantizer, further reducing the number of bits required for representation.
4. **Compression Ratio Variation:** We will examine the impact of varying the percentage of discarded coefficients on the compression ratio. Compression ratios of 50%, 75%, 90%, and 95% will be tested.
5. **Image Reconstruction:** The quantized coefficients are de-quantized and transformed back to the spatial domain using the inverse 2-D DCT, resulting in a reconstructed image.
6. **Evaluation:**
    - The total number of bits used to code the compressed image will be calculated.
    - The average number of bits used to represent each pixel (bits per pixel) will be determined.
    - The Peak Signal-to-Noise Ratio (PSNR) will be computed to quantify the quality of the reconstructed image compared to the original.

**Expected Outcome:**

By increasing the percentage of discarded coefficients, we expect to achieve higher compression ratios. However, this will likely come at the cost of reduced image quality, as reflected in the PSNR values. Analyzing the trade-off between compression ratio and PSNR will provide valuable insights into the effectiveness of this DCT-based image compression system.

I'd be glad to provide a formatted response combining the best aspects of previous responses and addressing potential issues, while adhering to GitHub Markdown formatting and avoiding code inclusion:

## Advanced Image Processing - Image Blending
### [Hybrid Images](https://github.com/MokshagnaRohit/Image-Processing-/blob/main/Image%20Blending/Hybrid_images.ipynb)
Take two images (of your choice), and try the hybrid images approach from the paper from Olivia et al http://olivalab.mit.edu/publications/OlivaTorralb_Hybrid_Siggraph06.pdf.

**Implementation:**

1. Select two images with complementary spatial frequency content (e.g., high-frequency facial details in one image and low-frequency scene information in another).
2. Compute the Fourier transform of both images.
3. Design a hybrid filter that assigns high-frequency content from one image and low-frequency content from the other image. This may involve using weighted averaging or other techniques.
4. Apply the hybrid filter to the respective magnitude components of the Fourier transforms.
5. Perform the inverse Fourier transform to reconstruct the hybrid image.
6. Display the original images and the resulting hybrid image.
  - Observe how the hybrid image combines visual elements from both source images.

**Note:**

- The specific implementation details will vary depending on the chosen libraries and filter design methods.
- Experiment with different filter shapes and parameters to explore their effects on image processing results.
- Reference the provided paper by Oliva et al. ([http://olivalab.mit.edu/hybridimage.htm](http://olivalab.mit.edu/hybridimage.htm)) for further details on hybrid image creation.

### [Multiresolution Blending using Gaussian/Laplacian Pyramids](https://github.com/MokshagnaRohit/Image-Processing-/blob/main/Image%20Blending/Blending.ipynb)

This section explores multiresolution blending, a technique for seamlessly merging multiple images. We'll reference the paper by Burt and Adelson ([http://ai.stanford.edu/~kosecka/burt-adelson-spline83.pdf](http://ai.stanford.edu/~kosecka/burt-adelson-spline83.pdf)) for theoretical background.

**Implementation:**

**(a) Image Selection and Blending Goal:**

1. **Choose Images:** Select two or more images for blending. Be creative! Consider factors like:
    - Desired visual effect (e.g., panoramic view from multiple images, combining object and background)
    - Complementary content (images with overlapping or related features)
    - Compatibility for seamless blending (e.g., similar lighting, perspective)

2. **Explain Purpose:** Describe the type of blending you aim to achieve (e.g., creating a panoramic view, enhancing an object) and why you believe multiresolution blending is a suitable approach for this specific case.

**(b) Gaussian and Laplacian Pyramid Construction:**

1. **Function Implementation:** Write a function `build_pyramids(image)` that takes an image and returns its Gaussian and Laplacian pyramids.
2. **Pyramid Depth:** Choose a suitable depth for the pyramids (e.g., 3-5 levels). Maintain the same depth for all pyramids in this section for consistency.
3. **Laplacian Reconstruction:** Verify that the Laplacian pyramid can be reconstructed back to the original image. This step confirms the correctness of your pyramid construction. You can use existing OpenCV functions where appropriate.

**(c) Mask Creation:**

1. **Mask Design:** Design binary masks that define the transition zones between the images you want to blend. Each mask will have a value of 1 in the region where the corresponding image should be visible and 0 in the blending area. 
2. **Multiple Images:** If blending more than two images, create additional masks to define transitions between each pair.

**(d) Direct Blending:**

1. **Function Implementation:** Write a function `direct_blend(image1, image2, mask)` that takes two images and a mask as input and performs direct blending using the formula:

   ```
   blended_image = (1 - mask) * image1 + mask * image2
   ```

2. **Visualization:** Apply the function to your chosen images and mask(s) to generate a directly blended image. Observe the visual result.

**(e) Alpha Blending:**

1. **Mask Smoothing:** Implement a function `blur_mask(mask)` that takes the mask and applies a Gaussian blur to soften the edges. This creates a smoother transition between the images in the blended result.
2. **Alpha Blending Function:** Write a function `alpha_blend(image1, image2, mask)` that performs alpha blending using the formula:

   ```
   blended_image = (1 - blur_mask(mask)) * image1 + blur_mask(mask) * image2
   ```

3. **Visualization:** Apply the alpha blending function to your images and the blurred mask(s). Analyze the result. Does the blended image appear more seamless, resembling a single entity (e.g., one fruit)?

**(f) Multiresolution Blending:**

1. **Function Implementation:** Write a function `multiblend(image1, image2, mask)` that performs multiresolution blending based on the algorithm discussed in class. The function should take the following inputs:
    - `image1`: First image to be blended
    - `image2`: Second image to be blended
    - `mask`: Mask defining the transition region
2. **Pyramid Construction:** Construct Gaussian pyramids for the mask and Laplacian pyramids for both images using the previously defined depth.
3. **Multiresolution Blending Algorithm:** Implement the multiresolution blending algorithm step-by-step within the function. This will involve blending corresponding levels of the pyramids and reconstructing the final blended image.
4. **Parameter Exploration:** Experiment with different parameters within the algorithm (e.g., weighting factors) to achieve a visually pleasing blend.

**Note:**

- The specific implementation details (code structure, library functions) will vary depending on your chosen programming language and tools.
- This breakdown provides a high-level structure for understanding and implementing multiresolution blending using Gaussian and Laplacian pyramids.

By following these steps and referencing the provided paper, you can implement multiresolution blending to create seamless image compositions.


This formatted response provides a concise guide for performing various Fourier domain image processing techniques in Python, while adhering to GitHub Markdown style and avoiding code inclusion. 

## Image Transformation and Manipulation

### [Fourier Domain Image Processing](https://github.com/MokshagnaRohit/Image-Processing-/tree/main/Fourier%20Domain)

This part explores various image processing techniques in the frequency domain using the Fourier transform.

**Implementation:**

**(a) Image Loading and Fourier Transform:**

1. Read an image of your choice using `cv2.imread()` (OpenCV) or `scipy.misc.imread()` (SciPy).
2. Convert the image to grayscale using `cv2.cvtColor()` (OpenCV) or `scipy.signal.rgb2gray()` (SciPy).
3. Compute the Fast Fourier Transform (FFT) using `cv2.dft()` (OpenCV) or `scipy.fft.fft2()` (SciPy).
4. Separate the magnitude and phase components using the absolute value and angle functions.
5. Plot the magnitude and phase using libraries like Matplotlib ([https://matplotlib.org/](https://matplotlib.org/)).

**(b) Frequency Domain Filtering:**

1. Design the desired filters (low-pass, high-pass, diagonal bandpass) in the frequency domain. This may involve creating masks or modifying the original Fourier spectrum.
2. Apply the filters to the magnitude component of the Fourier transform.
3. Set the phase component to zero for low-pass or high-pass filtering (optional for bandpass).
4. Perform the inverse Fast Fourier Transform (IFFT) using `cv2.idft()` (OpenCV) or `scipy.fft.ifft2()` (SciPy) to obtain the filtered image.
5. Plot the magnitude of the Fourier transform before and after filtering.
6. Display the original and filtered images for visual comparison.

**(c) Phase Swapping and Modification:**

1. Load two images of your choice.
2. Compute the Fourier transform of each image using the methods from step (a).
3. Separate the magnitude and phase components for both images.
4. Swap the phase components between the two images.
5. Perform the inverse Fourier transform to reconstruct the images with swapped phases.
6. Display the original images and the reconstructed images with swapped phases.
  - Observe how the visual content is potentially distorted while retaining some low-frequency information.

7. Experiment with modifying the phase information in different ways (e.g., scaling, adding noise) to explore its impact on the reconstructed image.
  - Analyze how these modifications affect the visual properties and potentially introduce artifacts.

### [2D Image Transformations and Feature Matching](https://github.com/MokshagnaRohit/Image-Processing-/tree/main/2D%20image%20transformations)

This section explores basic 2D image transformations (translation, rotation) and feature matching using custom detectors and descriptors. We'll utilize OpenCV for image manipulation and potentially reference external libraries like NumPy for numerical computations.

**Implementation:**

**(1) Image Creation, Translation, and Rotation:**

1. **Image Generation:** Create a 300x300 pixel image filled with black background using OpenCV's `cv2.zeros()` function.
2. **Quadrilateral Creation:** Draw a white irregular quadrilateral of approximately 50x50 pixels centered on the black background using drawing functions like `cv2.line()` or `cv2.rectangle()` with appropriate corner coordinates.
3. **Translation:** Apply a translation of (30, 100) to the image using `cv2.warpAffine()` or a similar function.
4. **Rotation:** Rotate the translated image by 45 degrees around its original center using `cv2.getRotationMatrix2D()` and `cv2.warpAffine()`.
5. **Display Transformed Image:** Display the final image with the rotated quadrilateral.

**(2) Custom Feature Detector and Descriptor:**

**Note:** This section emphasizes a custom approach. Built-in detectors/descriptors are not used here.

1. **Feature Detector Design:** Develop a simple feature detection algorithm to identify corner-like regions in the image. This could involve methods like:
    - **Harris Corner Detection:** Implement Harris corner detection from scratch, calculating corner response scores based on image gradients and windowing. Identify pixels with high response scores as potential corners.
    - **Simpler Corner Detection:** Explore alternative methods like finding points with high intensity gradients in two perpendicular directions, indicating potential corner locations.

2. **Descriptor Design:** Create a custom descriptor that captures the local image patch surrounding each detected corner. This could involve:
    - **Intensity Patch:** Extract a small square image patch centered on the corner as a descriptor, representing local intensity variations.
    - **Gradient Histogram:** Compute a local gradient histogram within the patch, capturing the distribution of edge orientations.

**(3) Feature Matching and Homography Estimation:**

1. **Matching Features:** Implement a feature matching algorithm to find corresponding corners between the original and transformed images using the custom descriptors. This may involve comparing descriptor distances (e.g., Euclidean distance) and selecting best matches based on a threshold.
2. **Subset Selection:** Choose a minimal number of reliable corner correspondences (e.g., four non-colinear points) for robust homography estimation.
3. **Homography Estimation:** Formulate a linear least squares system based on the corresponding points and solve for the homography matrix (combining translation, rotation, and scaling).

**Expected Outcome:**

- The image transformations and visualization demonstrate the applied translation and rotation.
- The custom feature detector and descriptor should identify and describe corner-like regions in the images.
- The feature matching and homography estimation should ideally recover the original translation and rotation parameters used in step (1).

**Note:**

- The specific implementation details (function definitions, library usage) will depend on your chosen programming language and tools.
- This breakdown provides a framework for exploring basic 2D transformations and feature matching with custom algorithms. Consider exploring existing feature detectors and descriptors (e.g., SIFT, SURF) for more sophisticated approaches.

### [Image Stitching with Feature Matching and RANSAC](https://github.com/MokshagnaRohit/Image-Processing-/tree/main/Image%20Stitching)

This section explores image stitching, a technique for creating panoramic views by combining multiple overlapping images. We'll utilize the SIFT (Scale-Invariant Feature Transform) algorithm for feature detection and matching, followed by RANSAC (RANdom SAmple Consensus) for robust homography estimation. OpenCV will be used for image processing and visualization.

**Implementation:**

**(1) Image Capture:**

1. Capture at least four overlapping images of a suitable scene using your phone camera.
2. Display the captured images in your report to visualize the scene being stitched.

**(2) Reference Image Selection:**

1. Choose one image as the reference frame. The middle image of the sequence is often preferred for minimal distortion, but any image can be chosen.
2. Discuss the challenges associated with choosing end images as the reference frame and potential solutions (e.g., composing homographies).

**(3) Feature Matching:**

1. **SIFT Feature Detection:** Use OpenCV's `cv2.xfeatures2d.SIFT_create()` function to detect SIFT features in each image.
2. **Descriptor Extraction:** Extract SIFT descriptors for each detected feature using `sift.compute()`.
3. **Pairwise Matching:** For each pair of images (excluding the reference image):
    - Compute pairwise Euclidean distances between all descriptor pairs.
    - Implement thresholding based on the ratio of the first and second nearest neighbor distances as described in Lowe's paper ([http://matthewalunbrown.com/papers/cvpr05.pdf](http://matthewalunbrown.com/papers/cvpr05.pdf)). Refer to Figure 6b for threshold selection.
    - Select feature pairs with a ratio below the threshold as tentative correspondences.
4. **Visualization:** Display a reasonable number of tentative feature correspondences between two images to visually assess the matching quality.

**(4) Homography Estimation with RANSAC:**

1. **RANSAC Algorithm:** For each image pair (excluding the reference image):
    - Use OpenCV's `cv2.findHomography()` function with the RANSAC flag enabled to estimate the homography matrix that relates the image to the reference frame.
    - Document the parameters used for `cv2.findHomography()`, such as the maximum reprojection error threshold and the number of RANSAC iterations.
2. **Homography Analysis:**
    - Report the estimated homography matrix for each image pair.
    - Calculate the reprojection error for each matched feature using the estimated homography.
    - Analyze the average and distribution of the reprojection errors to evaluate the accuracy of homography estimation.

**(5) Warping and Compositing:**

1. **Image Warping:** For each image (excluding the reference image):
    - Use OpenCV's `cv2.warpPerspective()` function with the estimated homography to warp the image into the reference frame's perspective.
2. **Mosaic Creation:** Combine the warped images and the reference image into a single panoramic mosaic image.
3. **Display Final Stitched Image:** Display the final stitched panorama to visualize the combined scene.

**Note:**

- The specific implementation details (function calls, parameter values) might vary slightly depending on the OpenCV version and coding style.
- This breakdown provides a framework for image stitching using SIFT feature matching and RANSAC-based homography estimation. Consider exploring advanced techniques like bundle adjustment for further refinement.

## [Pinspeck Cameras: Exploring Accidental Imaging](https://github.com/MokshagnaRohit/Image-Processing-/tree/main/Pinspeck%20Camera)

This section investigates pinspeck cameras, also known as anti-pinhole cameras, and their applications for extracting scene information. We'll reference the paper "Accidental Pinhole and Pinspeck Cameras" by Torralba and Freeman ([https://people.csail.mit.edu/torralba/research/accidentalcameras/](https://people.csail.mit.edu/torralba/research/accidentalcameras/)) and utilize OpenCV for video processing.

**Implementation**

**(a) Paper Review:**

Read and summarize the key points of the Torralba and Freeman paper:

- Understand the concept of pinspeck cameras and their characteristics.
- Explore the applications discussed in the paper (e.g., revealing hidden scenes).

**(b) Movie Loading:**

1. **Function Implementation:** Write a function `load_video(video_path)` that takes a video file path as input and uses `cv2.VideoCapture()` to read the video.
2. **Video Processing:** Within the function, iterate through each frame of the video for further processing.

**(c) Difference Video:**

1. **Reference Frame Selection:** Refer to the Torralba and Freeman paper for strategies on choosing a reference frame for background subtraction. You can choose:
    - The first frame (assuming static background initially).
    - The median of multiple frames to capture a more robust background representation.
2. **Difference Image Calculation:** For each frame (excluding the reference frame):
    - Calculate the absolute difference between the current frame and the reference frame using `cv2.absdiff()`.
3. **Visualization:** Display or save the resulting difference video, analyzing it for potential pinspeck camera effects (e.g., sudden appearance/disappearance of objects outside the window).

**(d) Pinspeck Region Isolation:**

1. **Manual Selection:** Identify a time segment in the difference video where changes outside the window suggest potential pinspeck camera effects.
2. **ROI Extraction:** Extract the corresponding frames from the original video that belong to the isolated segment.

**(e) Pinspeck Image Recreation:**

1. **Image Capture:** Using your camera or phone, capture two images:
    - An image of the background scene.
    - An image of the same scene with an occluder (e.g., your finger) partially covering the lens, creating a pinspeck effect.
2. **Difference Image:** Calculate the difference between the images with and without the occluder.
3. **Ground Truth:** Capture a clear image of the scene outside the window/pinspeck opening.
4. **Comparison:** Visually compare the difference image obtained from your pinspeck recreation with the ground truth scene outside the window. Observe if the pinspeck effect reveals any hidden information.

**Note:**

- The specific implementation details (function arguments, OpenCV parameter values) might vary depending on your coding style and chosen tools.
- This breakdown provides a framework for exploring pinspeck cameras using video processing techniques.
- Consider experimenting with different reference frame selection strategies and analyzing the impact on the difference video.

