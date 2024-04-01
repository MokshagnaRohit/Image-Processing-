# Image-Processing
This is a collection of  various Image Processing codes and projects I worked on. 

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

## [Wiener Filtering for Image Denoising](https://github.com/MokshagnaRohit/Image-Processing-/tree/main/Wiener%20Filtering)

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


