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

## [Wiener Filtering for Image Denoising](https://github.com/MokshagnaRohit/Image-Processing-/blob/master/Wiener%Filtering)

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


