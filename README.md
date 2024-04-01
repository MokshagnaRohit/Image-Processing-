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
