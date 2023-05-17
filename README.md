# 2D Convolution Using CPU/GPU

A simple cuda code for convolving 2D images with Gaussian kernels for a given sigma value.<br>
The code runs on Host (CPU) and Device (GPU) separately and saves the resulting images. On the Device, it runs both with and without shared memory to compare the execution time. 
All functions use separable convolution for optimization. <br>


### Input:<br>
* Image filename (TGA/TARGA format)
* Sigma value

### Output:<br>
* Blurred image (TGA/TARGA format) 
