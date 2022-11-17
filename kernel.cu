#include<iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include<string>
#include<vector>
#include<fstream>

# define PI           3.14159265358979323846  /* pi */

static void HandleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << "in" << file << "at line" << line;
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ )) 

// ------------------------------------------------------------------------------------------- //

// convolution along x on device
__global__ void dev_conv_x(char* out, char* img, float* kernel, int img_w, int out_h, int out_w, int K) {
    size_t i = blockDim.y * blockIdx.y + threadIdx.y;	// calculate row index, point to the output
    size_t j = blockDim.x * blockIdx.x + threadIdx.x;	// calculate column index, point to the output
    if (i >= out_h || j >= out_w) return;
    // initialize the register c
    float c[3];
    for (int x = 0; x < 3; x++)
        c[x] = 0.0f;
    
    // apply the convolution with Gaussian kernel along x axis
    for (int k = 0; k < K; k++) {
        c[0] += (unsigned char)img[3 * (i * img_w + j + k)] * kernel[k];
        c[1] += (unsigned char)img[3 * (i * img_w + j + k) + 1] * kernel[k];
        c[2] += (unsigned char)img[3 * (i * img_w + j + k) + 2] * kernel[k];
    }
    out[3 * (i * out_w + j)] = c[0];
    out[3 * (i * out_w + j) + 1] = c[1];
    out[3 * (i * out_w + j) + 2] = c[2];
}


__global__ void intensity(char* out, char* img, int img_w, int img_h) {
    size_t i = blockDim.y * blockIdx.y + threadIdx.y;	// calculate row index, point to the output
    size_t j = blockDim.x * blockIdx.x + threadIdx.x;	// calculate column index, point to the output
    if (i >= img_h || j >= img_w) return;
    out[3 * (i * img_w + j)] = 0.5 * (unsigned char)img[3 * (i * img_w + j)];
    out[3 * (i * img_w + j) + 1] = 0.5 * (unsigned char)img[3 * (i * img_w + j) + 1];
    out[3 * (i * img_w + j) + 2] = 0.5 * (unsigned char)img[3 * (i * img_w + j) + 2];
}




//  convolution along y on device
__global__ void dev_conv_y(char* out, char* img, float* kernel, int img_w, int out_h, int out_w, int K) {
    size_t i = blockDim.y * blockIdx.y + threadIdx.y;
    size_t j = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= out_h || j >= out_w) return;
    float c[3];
    for (int x{}; x < 3; x++)
        c[x] = 0.0f;
    for (int k{}; k < K; k++) {
        c[0] += (unsigned char)img[3 * ((i + k) * img_w + j)] * kernel[k];
        c[1] += (unsigned char)img[3 * ((i + k) * img_w + j) + 1] * kernel[k];
        c[2] += (unsigned char)img[3 * ((i + k) * img_w + j) + 2] * kernel[k];
    }
    out[3 * (i * out_w + j)] = c[0];
    out[3 * (i * out_w + j) + 1] = c[1];
    out[3 * (i * out_w + j) + 2] = c[2];
}


// ------------------------------------------------------------------------------------------- //


// convolution kernel along x running on host
void host_conv_x(char* out, char* img, float* kernel, int img_w, int out_h, int out_w, int K) {
    float c[3];
    for (int i = 0; i < out_h; i++) {										// calculate the i (row) index, point to the outMatrix
        for (int j = 0; j < out_w; j++) {									// calculate the j (column) index, point to the outMatrix
            // initialize the register c to store the results
            for (int x{}; x < 3; x++)
                c[x] = 0.0f;
            // convolving with Gaussian kernel along x axis
            for (int k{}; k < K; k++) {
                c[0] += (unsigned char)img[3 * (i * img_w + j + k)] * kernel[k];
                c[1] += (unsigned char)img[3 * (i * img_w + j + k) + 1] * kernel[k];
                c[2] += (unsigned char)img[3 * (i * img_w + j + k) + 2] * kernel[k];
            }
            out[3 * (i * out_w + j)] = c[0];
            out[3 * (i * out_w + j) + 1] = c[1];
            out[3 * (i * out_w + j) + 2] = c[2];
        }
    }
}

// convolution kernel along y running on host, same as convolution_x
void host_conv_y(char* out, char* img, float* kernel, int inp_w, int out_h, int out_w, int K) {
    float c[3];
    for (int j = 0; j < out_w; j++) {
        for (int i = 0; i < out_h; i++) {
            for (int x = 0; x < 3; x++)
                c[x] = 0.0f;
            for (int k = 0; k < K; k++) {
                c[0] += (unsigned char)img[3 * ((i + k) * inp_w + j)] * kernel[k];
                c[1] += (unsigned char)img[3 * ((i + k) * inp_w + j) + 1] * kernel[k];
                c[2] += (unsigned char)img[3 * ((i + k) * inp_w + j) + 2] * kernel[k];
            }
            out[3 * (i * out_w + j)] = c[0];
            out[3 * (i * out_w + j) + 1] = c[1];
            out[3 * (i * out_w + j) + 2] = c[2];
        }
    }
}


// ------------------------------------------------------------------------------------------- //


void write_tga(std::string filename, char* bytes, int width, int height) {
    std::ofstream outfile;
    outfile.open(filename, std::ios::binary | std::ios::out);	// open a binary file
    outfile.put(0);						// id length (field 1)
    outfile.put(0);						// color map type (field 2)
    outfile.put(2);						// image_type (field 3)
    outfile.put(0); outfile.put(0);		// color map field entry index (field 4)
    outfile.put(0); outfile.put(0);		// color map length (field 4)
    outfile.put(0);				// color map entry size (field 4)
    outfile.put(0); outfile.put(0);		// x origin (field 5)
    outfile.put(0); outfile.put(0);		// y origin (field 5)
    outfile.write((char*)&width, 2);		// image width (field 5)
    outfile.write((char*)&height, 2);		// image height (field 5)
    outfile.put(24);				// pixel depth (field 5)
    outfile.put(0);				// image descriptor (field 5)
    outfile.write(bytes, width * height * 3);		// write the image data
    outfile.close();				// close the file
}

std::vector<char> read_tga(std::string filename, int& width, int& height) {
    std::ifstream infile;
    infile.open(filename, std::ios::binary | std::ios::out);        // open the file for binary writing
    if (!infile.is_open()) {
        std::cout << "ERROR: Unable to open file " << filename << std::endl;
        return std::vector<char>();
    }
    char id_length;                                infile.get(id_length);                            // id length (field 1)
    char cmap_type;                                infile.get(cmap_type);                            // color map type (field 2)
    char image_type;                            infile.get(image_type);                        // image_type (field 3)
    char field_entry_a, field_entry_b;
    infile.get(field_entry_a);                infile.get(field_entry_b);                        // color map field entry index (field 4)
    char map_length_a, map_length_b;
    infile.get(map_length_a);                infile.get(map_length_b);                        // color map field entry index (field 4)
    char map_size;                                infile.get(map_size);                            // color map entry size (field 4)
    char origin_x_a, origin_x_b;
    infile.get(origin_x_a);                infile.get(origin_x_b);                        // x origin (field 5)
    char origin_y_a, origin_y_b;
    infile.get(origin_y_a);                infile.get(origin_y_b);                        // x origin (field 5)

    infile.read((char*)&width, 2);
    infile.read((char*)&height, 2);
    char pixel_depth;                            infile.get(pixel_depth);
    char descriptor;                            infile.get(descriptor);

    std::vector<char> bytes(width * height * 3);
    infile.read(&bytes[0], width * height * 3);

    infile.close();                    // close the file

    return bytes;
}

int main(int argc, char* argv[]) {
    //std::cout << "Convolve image.tga 40";
    if (argc != 3) {
        fprintf(stderr, "Error: 3 parameters expected. Found %d\n", argc);
        return 1;
    }

    std::string filename(argv[1]);
    int sigma = atoi(argv[2]);

    std::cout << "Filename: " << filename << std::endl;
    std::cout << "Input Sigma: " << sigma << std::endl;
    
    //Calculating kernel size
    int k_size = 4 * sigma;                          //radius = 2 * sigma
                                                     //kernel size = 2 * radius + 1
    if (k_size % 2 == 0) k_size++;                   //kernel size should be odd
    float miu = k_size / 2;

    int width = 0;		
    int height = 0;		
    int size = 0;		// size of the image array
    // No padding
    // strides = 1

    // loading image into a char* array
    std::cout << "Loading the image..." << std::endl;
    std::vector<char> imageVector = read_tga(filename, width, height);
    std::cout << "Loading finished.\n" <<  "width: " << width << " height: " << height << std::endl;
    size = width * height * 3;
    char* imageArray = &imageVector[0];

    // array sizes after convolution along x axis
    int x_height = height;
    int x_width = width - k_size + 1;
    int x_size = x_height * x_width * 3;
    char* x_output = (char*)malloc(x_size * sizeof(char));
    std::cout << "Part 1 finished." << std::endl;

    // array sizes after convolution along y axis
    int y_height = x_height - k_size + 1;
    int y_width = x_width;
    int y_size = y_height * y_width * 3;
    char* y_output = (char*)malloc(y_size * sizeof(char));
    std::cout << "Part 2 finished." << std::endl;

    // gaussian kernel as a float*
    float* gKernel = (float*)malloc(k_size * sizeof(float));
    int s = 2 * sigma * sigma;
    for (int i = 0; i < k_size; i++) {
        gKernel[i] = 1 / sqrt(s * (float)PI) * exp(-(i - miu)*(i - miu) / s);
        //std::cout << gKernel[i] << " | ";
        //return 0;
    }



    // running on host or device starts
    int device{};
    std::cout << "Host(1) or device(0)?" << std::endl;
    std::cin >> device;

    if (device) {
        // -------------------------------------- CPU ---------------------------------------- //

        std::cout << "------------------------- CPU version -------------------------" << std::endl;
        std::cout << "Convolving on HOST..." << std::endl;

        clock_t start, finish;
        start = clock();

        // convolving along x
        host_conv_x(x_output, imageArray, gKernel, width, x_height, x_width, k_size);
        // convolving along y 
        host_conv_y(y_output, x_output, gKernel, x_width, y_height, y_width, k_size);

        finish = clock();  // time finishs
        std::cout << "It takes " << (double)(finish - start) / CLOCKS_PER_SEC << " s to convolve on CPU" << std::endl;
        write_tga("out.tga", y_output, y_width, y_height);
        std::cout << "Convolution on CPU finished" << std::endl;
        //// -------------------------------------- CPU ---------------------------------------- //
    }


    else {
        //// -------------------------------------- GPU ---------------------------------------- //

        std::cout << "------------------------- GPU version -------------------------" << std::endl;
        int d;
        HANDLE_ERROR(cudaGetDevice(&d));
        std::cout << "Current device: " << d << std::endl;
        cudaDeviceProp prop;
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, d));
        
        float* gpu_gKernel;
        char* gpu_image;
        char* gpu_output_x;
        char* gpu_output_y;
        char* gpu_image_i;
        std::cout << "Image size: " << size << std::endl;

        // allocate memory for image, kernel, and two convoled outputs
        //HANDLE_ERROR(cudaMalloc(&gpu_gKernel, k_size * sizeof(float)));
        HANDLE_ERROR(cudaMalloc(&gpu_image, size * sizeof(char)));
        //HANDLE_ERROR(cudaMalloc(&gpu_output_x, x_size * sizeof(char)));
        //HANDLE_ERROR(cudaMalloc(&gpu_output_y, y_size * sizeof(char))); 
        HANDLE_ERROR(cudaMalloc(&gpu_image_i, size * sizeof(char)));
        // copy image and kernel from main memory to Device
        HANDLE_ERROR(cudaMemcpy(gpu_image, imageArray, size * sizeof(char), cudaMemcpyHostToDevice));
        //HANDLE_ERROR(cudaMemcpy(gpu_gKernel, gKernel, k_size * sizeof(float), cudaMemcpyHostToDevice));
        
        size_t blockDim = sqrt(prop.maxThreadsPerBlock);
        dim3 threads(blockDim, blockDim);
        std::cout << "threads.x: " << threads.x << ",\tthreads.y: " << threads.y << std::endl;
        dim3 blocks(width / threads.x + 1, height / threads.y + 1);
        char* imgoutput = (char*)malloc(size * sizeof(char));

        // starting GPU timer
        cudaEvent_t g_start;
        cudaEvent_t g_stop;
        cudaEventCreate(&g_start);
        cudaEventCreate(&g_stop);
        cudaEventRecord(g_start, NULL);


        std::cout << "Convolving on DEVICE..." << std::endl;
        // convolving along x
        //dev_conv_x <<< blocks, threads >>> (gpu_output_x, gpu_image, gpu_gKernel, width, x_height, x_width, k_size);
        // convolving along y
        //dev_conv_y <<< blocks, threads >>> (gpu_output_y, gpu_output_x, gpu_gKernel, x_width, y_height, y_width, k_size);

        intensity << < blocks, threads >> > (gpu_image_i, gpu_image, width, height);

        // copy convolved outputs from Device to main memory
        //HANDLE_ERROR(cudaMemcpy(x_output, gpu_output_x, x_size * sizeof(char), cudaMemcpyDeviceToHost));
        //HANDLE_ERROR(cudaMemcpy(y_output, gpu_output_y, y_size * sizeof(char), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(imgoutput, gpu_image_i, size * sizeof(char), cudaMemcpyDeviceToHost));
        // timer ends
        cudaEventRecord(g_stop, NULL);
        cudaEventSynchronize(g_stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, g_start, g_stop);
        std::cout << "It takes " << elapsedTime << " ms to convolve on GPU" << std::endl;

        // output file
        //write_tga("out_GPU_x.tga", x_output, x_width, x_height);
        write_tga("out_GPU.tga", imgoutput, width, height);

        std::cout << "Convolution on GPU finished" << std::endl;

        //// -------------------------------------- GPU ---------------------------------------- //
    }
}