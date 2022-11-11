#include<iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
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


// convolution kernel along x running on device (without shared memory)
__global__ void kernelConvolution_x_ns(char* C, char* A, float* B, int M, int N, int M_x, int N_x, int K) {
    size_t i = blockDim.y * blockIdx.y + threadIdx.y;	// calculate the i (row) index, point to the outMatrix
    size_t j = blockDim.x * blockIdx.x + threadIdx.x;	// calculate the j (column) index, point to the outMatrix
    if (i >= M_x || j >= N_x) return;
    // initialize the register c to store the results
    float c[3];
    for (int x = 0; x < 3; x++) {
        c[x] = 0;
    }
    //kernelConvolution_x_ns<<<blocks, threads>>>(gpu_outArray_x, gpu_inArray, gpu_gKernel, height, width, height_x, width_x, k_size);
    // apply the convolution with Gaussian kernel along x axis
    for (int k = 0; k < K; k++) {
        c[0] += (float)A[3 * (i * N + j + k)] * B[k];
        c[1] += (float)A[3 * (i * N + j + k) + 1] * B[k];
        c[2] += (float)A[3 * (i * N + j + k) + 2] * B[k];
    }
    // copy results from register to outMatrix
    C[3 * (i * N_x + j)] = (unsigned char)c[0];
    C[3 * (i * N_x + j) + 1] = (unsigned char)c[1];
    C[3 * (i * N_x + j) + 2] = (unsigned char)c[2];
}

// convolution kernel along y running on device (without shared memory), same as kernelConvolution_x_ns
__global__ void kernelConvolution_y_ns(char* C, char* A, float* B, int M, int N, int M_y, int N_y, int K) {
    size_t i = blockDim.y * blockIdx.y + threadIdx.y;
    size_t j = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= M_y || j >= N_y) return;
    float c[3];
    for (int x = 0; x < 3; x++) {
        c[x] = 0;
    }
    for (int k = 0; k < K; k++) {
        c[0] += (float)A[3 * ((i + k) * N + j)] * B[k];
        c[1] += (float)A[3 * ((i + k) * N + j) + 1] * B[k];
        c[2] += (float)A[3 * ((i + k) * N + j) + 2] * B[k];
    }
    C[3 * (i * N_y + j)] = (unsigned char)c[0];
    C[3 * (i * N_y + j) + 1] = (unsigned char)c[1];
    C[3 * (i * N_y + j) + 2] = (unsigned char)c[2];
}


// ------------------------------------------------------------------------------------------- //


// convolution kernel along x running on host

void convolution_x(char* C, char* A, float* B, int M, int N, int M_x, int N_x, int K) {
    float c[3];
    for (int i = 0; i < M_x; i++) {										// calculate the i (row) index, point to the outMatrix
        for (int j = 0; j < N_x; j++) {									// calculate the j (column) index, point to the outMatrix
            // initialize the register c to store the results
            for (int x = 0; x < 3; x++) {
                c[x] = 0;
            }
            // apply the convolution with Gaussian kernel along x axis
            for (int k = 0; k < K; k++) {
                c[0] += (float)A[3 * (i * N + j + k)] * B[k];
                c[1] += (float)A[3 * (i * N + j + k) + 1] * B[k];
                c[2] += (float)A[3 * (i * N + j + k) + 2] * B[k];
            }
            // copy results from register to outMatrix
            C[3 * (i * N_x + j)] = (unsigned char)c[0];
            C[3 * (i * N_x + j) + 1] = (unsigned char)c[1];
            C[3 * (i * N_x + j) + 2] = (unsigned char)c[2];
        }
    }
}

// convolution kernel along y running on host, same as convolution_x
void convolution_y(char* C, char* A, float* B, int M, int N, int M_y, int N_y, int K) {
    float c[3];
    for (int j = 0; j < N_y; j++) {
        for (int i = 0; i < M_y; i++) {
            for (int x = 0; x < 3; x++) {
                c[x] = 0;
            }
            for (int k = 0; k < K; k++) {
                c[0] += (float)A[3 * ((i + k) * N + j)] * B[k];
                c[1] += (float)A[3 * ((i + k) * N + j) + 1] * B[k];
                c[2] += (float)A[3 * ((i + k) * N + j) + 2] * B[k];
            }
            C[3 * (i * N_y + j)] = (unsigned char)c[0];
            C[3 * (i * N_y + j) + 1] = (unsigned char)c[1];
            C[3 * (i * N_y + j) + 2] = (unsigned char)c[2];
        }
    }
}


// ------------------------------------------------------------------------------------------- //


void write_tga(std::string filename, char* bytes, short width, short height) {
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

std::vector<char> read_tga(std::string filename, short& width, short& height) {
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

    std::cout << "filename: " << filename << std::endl;
    std::cout << "sigma: " << sigma << std::endl;

    int k_size = 6 * sigma;  //calculate k
    if (k_size % 2 == 0) k_size++; //make sure k is odd
    float miu = k_size / 2;

    //char pSix[10];		// indicates this is a PPM image
    short width = 0;		// width of the image
    short height = 0;		// height of the image
    int maximum = 0;	// maximum pixel value
    int size = 0;		// size of the array


    // read the PPM file and store its contents inside an array and return the pointer to that array to pixelArray
    // notice the type of the array is unsigned char!!!!
    std::cout << "Starts loading." << std::endl;
    std::vector<char> imageVector = read_tga(filename, width, height);
    std::cout << "Load finished." << std::endl;
    std::cout << "width: " << width << " height: " << height << std::endl;
    size = width * height * 3;
    char* imageArray = &imageVector[0];

    //char* outArray_x = (char*)malloc(size_x * sizeof(char));

    // allocate output array for pixels after convolution along x axis
    int height_x = height;
    int width_x = (width - k_size) + 1;
    int size_x = height_x * width_x * 3;
    char* outArray_x = (char*)malloc(size_x * sizeof(char));
    std::cout << "Part 1 finished." << std::endl;
    // allocate output array for pixels after convolution along y axis
    int height_y = (height_x - k_size) + 1;
    int width_y = width_x;
    int size_y = height_y * width_y * 3;
    char* outArray_y = (char*)malloc(size_y * sizeof(char));
    std::cout << "Part 2 finished." << std::endl;
    // define a float pointer to the gaussian kernel
    float* gKernel = (float*)malloc(k_size * sizeof(float));
    for (int i = 0; i < k_size; i++) {
        gKernel[i] = 1 / sqrt(2 * sigma * sigma * (float)PI) * exp(-(i - miu)*(i - miu) / (2 * sigma * sigma));
        //return 0;
    }

    std::cout << "Part 3 finished." << std::endl;
    // -------------------------------------- CPU VERSION ---------------------------------------- //

    std::cout << "--------------------- CPU version ---------------------" << std::endl;
    std::cout << "Starts doing convolution on CPU" << std::endl;

    clock_t start, finish;	//Create timer
    double duration;
    start = clock();  // time starts

    // do kernel convolution along x axis
    convolution_x(outArray_x, imageArray, gKernel, height, width, height_x, width_x, k_size);
    // do kernel convolution along y axis
    convolution_y(outArray_y, outArray_x, gKernel, height_x, width_x, height_y, width_y, k_size);

    finish = clock();  // time ends
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    std::cout << "It takes " << duration << " s to do the CPU based convolution!" << std::endl;

    //// output the ppm after convolution along x axis
    ////char outFile_x[50] = "./src/cpu_out_x.ppm";
    write_tga("outy.tga", outArray_y, width, height);
    ////writePPM(outFile_x, outArray_x, pSix, &width_x, &height_x, &maximum, &size_x);
    //// output the ppm after convolution along x and y axis
    ////char outFile[50] = "./src/cpu_out.ppm";
    ////writePPM(outFile, outArray_y, pSix, &width_y, &height_y, &maximum, &size_y);

    std::cout << "Convolution on CPU finished" << std::endl;
    //149.525
    // -------------------------------------- CPU VERSION ---------------------------------------- //

    // -------------------------------------- GPU VERSION ---------------------------------------- //

    std::cout << "--------------------- GPU version ---------------------" << std::endl;
    cudaDeviceProp props;																//declare a CUDA properties structure
    HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));									//get the properties of the first CUDA device

    float* gpu_gKernel;																				//pointer to the gaussian kernel
    char* gpu_inArray;  																	//pointer to input Array
    char* gpu_outArray_x;  																//pointer to output Array after convolution along x
    char* gpu_outArray_y;  																//pointer to output Array after all the convolution

    std::cout << "size: " << size << std::endl;

    HANDLE_ERROR(cudaMalloc(&gpu_gKernel, k_size * sizeof(float)));  							//allocate memory on device
    HANDLE_ERROR(cudaMalloc(&gpu_inArray, size * sizeof(char)));  							    //allocate memory on device
    HANDLE_ERROR(cudaMalloc(&gpu_outArray_x, size_x * sizeof(char)));  							//allocate memory on device
    HANDLE_ERROR(cudaMalloc(&gpu_outArray_y, size_y * sizeof(char)));  							//allocate memory on device

    HANDLE_ERROR(cudaMemcpy(gpu_gKernel, gKernel, k_size * sizeof(float), cudaMemcpyHostToDevice));  //copy the array from main memory to device
    HANDLE_ERROR(cudaMemcpy(gpu_inArray, imageArray, size * sizeof(char), cudaMemcpyHostToDevice));     //copy the array from main memory to device

    size_t blockDim = sqrt(props.maxThreadsPerBlock);
    dim3 threads(blockDim, blockDim);
    std::cout << "threads.x: " << threads.x << std::endl;
    std::cout << "threads.y: " << threads.y << std::endl;
    dim3 blocks(width / threads.x + 1, height / threads.y + 1);


    // without shared memory
    std::cout << "Starts doing convolution on GPU without shared memory" << std::endl;

    //	utilize cudaEvent_t to serve as GPU timer
    cudaEvent_t d_start;
    cudaEvent_t d_stop;
    cudaEventCreate(&d_start);
    cudaEventCreate(&d_stop);
    cudaEventRecord(d_start, NULL);

    kernelConvolution_x_ns <<<blocks, threads >> > (gpu_outArray_x, gpu_inArray, gpu_gKernel, height, width, height_x, width_x, k_size);
    kernelConvolution_y_ns <<<blocks, threads >> > (gpu_outArray_y, gpu_outArray_x, gpu_gKernel, height_x, width_x, height_y, width_y, k_size);

    HANDLE_ERROR(cudaMemcpy(outArray_x, gpu_outArray_x, size_x * sizeof(char), cudaMemcpyDeviceToHost));  //copy the array back from device to main memory
    HANDLE_ERROR(cudaMemcpy(outArray_y, gpu_outArray_y, size_y * sizeof(char), cudaMemcpyDeviceToHost));  //copy the array back from device to main memory

    //	end of cudaEvent_t, calculate the time and show
    cudaEventRecord(d_stop, NULL);
    cudaEventSynchronize(d_stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, d_start, d_stop);
    std::cout << "It takes " << elapsedTime << " ms to do the GPU based convolution!" << std::endl;

    // output ppm file
    char outFile_gpu_x_ns[50] = "./src/gpu_out_x_ns.ppm";
    write_tga("out_GPU_x.tga", outArray_x, width_x, height_x);
    char outFile_gpu_ns[50] = "./src/gpu_out_ns.ppm";
    write_tga("out_GPU_y.tga", outArray_x, width_x, height_x);

    std::cout << "Convolution on GPU without shared memory finished" << std::endl;

}