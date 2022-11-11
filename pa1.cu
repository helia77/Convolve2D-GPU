#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

#define PI 3.14159

static void HandleError( cudaError_t err, const char *file, int line ) {
	if (err != cudaSuccess) {
		cout<<cudaGetErrorString( err )<<"in"<< file <<"at line"<< line;
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

// convolution kernel along x running on device (with shared memory)
__global__ void kernelConvolution_x(unsigned char* C, unsigned char* A, float* B, int M, int N, int M_x, int N_x, int K){
	extern __shared__ unsigned char sharedPtr[];
	size_t i = blockDim.y * blockIdx.y + threadIdx.y;   // calculate the i (row) index, point to the outMatrix
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;   // calculate the j (column) index, point to the outMatrix
	size_t ti = threadIdx.y;							// calculate the i (row) index, point to the shared memory
	size_t tj = threadIdx.x;							// calculate the j (column) index, point to the shared memory
	if (i >= M_x || j >= N_x) return;
	int L = blockDim.x + K - 1;							// length of data required in one block
	// copy data to shared memory
	for (int l = tj; l < L; l += blockDim.x){
		sharedPtr[3 * (ti * L + l)] = A[3 * (i * N + blockDim.x * blockIdx.x + l)];
		sharedPtr[3 * (ti * L + l) + 1] = A[3 * (i * N + blockDim.x * blockIdx.x + l) + 1];
		sharedPtr[3 * (ti * L + l) + 2] = A[3 * (i * N + blockDim.x * blockIdx.x + l) + 2];
	}
	__syncthreads();

	// initialize the register c to store the results
	float c[3];
	for (int x = 0; x < 3; x++){
		c[x] = 0;
	}
	// apply the convolution with Gaussian kernel along x axis
	for (int k = 0; k < K; k++){
		c[0] += (float)sharedPtr[3 * (ti * L + tj + k)] * B[k];
		c[1] += (float)sharedPtr[3 * (ti * L + tj + k) + 1] * B[k];
		c[2] += (float)sharedPtr[3 * (ti * L + tj + k) + 2] * B[k];
	}
	// copy results from register to outMatrix
	C[3 * (i * N_x + j)] = (unsigned char)c[0];
	C[3 * (i * N_x + j) + 1] = (unsigned char)c[1];
	C[3 * (i * N_x + j) + 2] = (unsigned char)c[2];
}

// convolution kernel along y running on device (with shared memory), same as kernelConvolution_x
__global__ void kernelConvolution_y(unsigned char* C, unsigned char* A, float* B, int M, int N, int M_y, int N_y, int K){
	extern __shared__ unsigned char sharedPtr[];
	size_t i = blockDim.y * blockIdx.y + threadIdx.y;
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;
	size_t ti = threadIdx.y;
	size_t tj = threadIdx.x;
	if (i >= M_y || j >= N_y) return;
	int L = blockDim.y + K - 1;
	for (int l = ti; l < L; l += blockDim.y){
		sharedPtr[3 * (l * blockDim.x + tj)] = A[3 * ((blockDim.y * blockIdx.y + l) * N + j)];
		sharedPtr[3 * (l * blockDim.x + tj) + 1] = A[3 * ((blockDim.y * blockIdx.y + l) * N + j) + 1];
		sharedPtr[3 * (l * blockDim.x + tj) + 2] = A[3 * ((blockDim.y * blockIdx.y + l) * N + j) + 2];
	}
	__syncthreads();

	float c[3];
	for (int x = 0; x < 3; x++){
		c[x] = 0;
	}
	for (int k = 0; k < K; k++){
		c[0] += (float)sharedPtr[3 * ((ti + k) * blockDim.x + tj)] * B[k];
		c[1] += (float)sharedPtr[3 * ((ti + k) * blockDim.x + tj) + 1] * B[k];
		c[2] += (float)sharedPtr[3 * ((ti + k) * blockDim.x + tj) + 2] * B[k];
	}
	C[3 * (i * N_y + j)] = (unsigned char)c[0];
	C[3 * (i * N_y + j) + 1] = (unsigned char)c[1];
	C[3 * (i * N_y + j) + 2] = (unsigned char)c[2];
}

// convolution kernel along x running on device (without shared memory)
__global__ void kernelConvolution_x_ns(unsigned char* C, unsigned char* A, float* B, int M, int N, int M_x, int N_x, int K){
	size_t i = blockDim.y * blockIdx.y + threadIdx.y;	// calculate the i (row) index, point to the outMatrix
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;	// calculate the j (column) index, point to the outMatrix
	if (i >= M_x || j >= N_x) return;
	// initialize the register c to store the results
	float c[3];
	for (int x = 0; x < 3; x++){
		c[x] = 0;
	}
	// apply the convolution with Gaussian kernel along x axis
	for (int k = 0; k < K; k++){
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
__global__ void kernelConvolution_y_ns(unsigned char* C, unsigned char* A, float* B, int M, int N, int M_y, int N_y, int K){
	size_t i = blockDim.y * blockIdx.y + threadIdx.y;
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= M_y || j >= N_y) return;
	float c[3];
	for (int x = 0; x < 3; x++){
		c[x] = 0;
	}
	for (int k = 0; k < K; k++){
		c[0] += (float)A[3 * ((i + k) * N + j)] * B[k];
		c[1] += (float)A[3 * ((i + k) * N + j) + 1] * B[k];
		c[2] += (float)A[3 * ((i + k) * N + j) + 2] * B[k];
	}
	C[3 * (i * N_y + j)] = (unsigned char)c[0];
	C[3 * (i * N_y + j) + 1] = (unsigned char)c[1];
	C[3 * (i * N_y + j) + 2] = (unsigned char)c[2];
}

// convolution kernel along x running on host
void convolution_x(unsigned char* C, unsigned char* A, float* B, int M, int N, int M_x, int N_x, int K){
	float c[3];
	for (int i = 0; i < M_x; i++){										// calculate the i (row) index, point to the outMatrix
		for (int j = 0; j < N_x; j++){									// calculate the j (column) index, point to the outMatrix
			// initialize the register c to store the results
			for (int x = 0; x < 3; x++){
				c[x] = 0;
			}
			// apply the convolution with Gaussian kernel along x axis
			for (int k = 0; k < K; k++){
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

// convolution kernel along x running on host, same as convolution_x
void convolution_y(unsigned char* C, unsigned char* A, float* B, int M, int N, int M_y, int N_y, int K){
	float c[3];
	for (int j = 0; j < N_y; j++){
		for (int i = 0; i < M_y; i++){
			for (int x = 0; x < 3; x++){
				c[x] = 0;
			}
			for (int k = 0; k < K; k++){
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
// This readPPM function is found online at http://www.cplusplus.com/forum/general/208835/
// As I didn't know how to use the format information about PPM file and read it in C++, I take
// this online solution as the reference. However this original function cannot read the pixelArray
// because the argument for fread function (line 178) is incorrect. I solved that.
unsigned char* readPPM(const char* fileName, char* pSix, int* width, int* height, int* maximum, int* size) {

	// open the file to read just the header reading
	FILE* fr = fopen(fileName, "r");

	// formatted read of header
	fscanf(fr, "%s\n", pSix);    // "\n" is missing in the original function

	// check to see if it's a PPM image file
	if (strncmp(pSix, "P6" , 10) != 0) {
		printf("They are not the same\n");
	} else {
		printf("They are the same\n");
	}

	// skip the line start with '#'
	// !!Double quotes are the shortcut syntax for a c-string in C++. If you want to compare a single character, you must use single quotes instead.
	char line[100];
	while (fgets (line , sizeof(line) , fr) != NULL ){
		if (line[0] == '#') continue;
		else break;
//		puts(line);
	}

	// read the rest of header
	sscanf(line, "%d\n %d\n", width, height);
    fscanf(fr, "%d\n", maximum);

    // check to see if they were stored properly
    printf("PSix: %s\n", pSix);
    printf("Width: %d\n", *width);
    printf("Height: %d\n", *height);
    printf("maximum: %d\n", *maximum);

    *size = *width * *height * 3;

    // allocate array for pixels
    unsigned char* pixels = (unsigned char*) malloc(*size * sizeof(char));   // equivalent to unsigned char* pixels = new unsigned char[size];

	// unformatted read of binary pixel data
    fread(pixels, sizeof(char), *size, fr);

	// close file
	fclose(fr);

	// return the array
	return pixels;
}
// ------------------------------------------------------------------------------------------- //

void writePPM(const char* fileName, unsigned char* pixelArray, char* pSix, int* width, int* height, int* maximum, int* size){
	// open the file to write
	FILE* fr = fopen(fileName, "w");
	// read the header
	fprintf(fr, "%s\n", pSix);
	fprintf(fr, "%d %d\n", *width, *height);
	fprintf(fr, "%d\n", *maximum);
	// read the rarray
	fwrite(pixelArray, sizeof(char), *size, fr);
	// close file
	fclose(fr);
}



int main(int argc, char* argv[]){							//main function											//GPU console output

	if(argc != 3)       //there should be three arguments
		return 1;       //exit and return an error
//	string filename(argv[1]);
	char* fileName = argv[1];
	int sigma = atoi(argv[2]);
	cout << "filename: " << fileName << endl;
	cout << "sigma: " << sigma << endl;
	cout << "------------------------------------------------------" << endl;
//	int sigma = 5;
	int k_size = 6 * sigma;  //calculate k
	if(k_size % 2 == 0) k_size++; //make sure k is odd
	float miu = k_size / 2;

//	char fileName[50] = "Galaxy.ppm";
//	char fileName[50] = "./src/Aerial.512.ppm";
	char pSix[10];		// indicates this is a PPM image
	int width = 0;		// width of the image
	int height = 0;		// height of the image
	int maximum = 0;	// maximum pixel value
	int size = 0;		// size of the array

	// read the PPM file and store its contents inside an array and return the pointer to that array to pixelArray
	// notice the type of the array is unsigned char!!!!
	cout << "Starts loading." << endl;
	unsigned char* inArray = readPPM(fileName, pSix, &width, &height, &maximum, &size);
	cout << "Load finished." << endl;

	// allocate output array for pixels after convolution along x axis
 	int height_x = height;
 	int width_x = (width - k_size) + 1;
 	int size_x =  height_x * width_x * 3;
 	unsigned char* outArray_x = (unsigned char*) malloc(size_x * sizeof(char));

 	// allocate output array for pixels after convolution along y axis
 	int height_y = (height_x - k_size) + 1;
 	int width_y = width_x;
 	int size_y = height_y * width_y * 3;
 	unsigned char* outArray_y = (unsigned char*) malloc(size_y * sizeof(char));

 	// define a float pointer to the gaussian kernel
 	float* gKernel = (float*) malloc(k_size * sizeof(float));
 	for (int i = 0; i < k_size; i++){
 		gKernel[i] = 1 / sqrt(2 * sigma * sigma * (float)PI) * exp(- pow((i - miu), 2) / (2 * sigma * sigma));
 	}

 	// -------------------------------------- CPU VERSION ---------------------------------------- //

	cout << "--------------------- CPU version ---------------------" << endl;
	cout << "Starts doing convolution on CPU" << endl;

	clock_t start, finish;	//Create timer
	double duration;
	start = clock();  // time starts

	// do kernel convolution along x axis
 	convolution_x(outArray_x, inArray, gKernel, height, width, height_x, width_x, k_size);
 	// do kernel convolution along y axis
 	convolution_y(outArray_y, outArray_x, gKernel, height_x, width_x, height_y, width_y, k_size);

	finish = clock();  // time ends
	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	cout << "It takes " <<duration <<" s to do the CPU based convolution!" << endl;

 	// output the ppm after convolution along x axis
 	char outFile_x[50] = "./src/cpu_out_x.ppm";
 	writePPM(outFile_x, outArray_x, pSix, &width_x, &height_x, &maximum, &size_x);
 	// output the ppm after convolution along x and y axis
 	char outFile[50] = "./src/cpu_out.ppm";
 	writePPM(outFile, outArray_y, pSix, &width_y, &height_y, &maximum, &size_y);

 	cout << "Convolution on CPU finished" << endl;

	// -------------------------------------- CPU VERSION ---------------------------------------- //


	// -------------------------------------- GPU VERSION ---------------------------------------- //

 	cout << "--------------------- GPU version ---------------------" << endl;
 	cudaDeviceProp props;																//declare a CUDA properties structure
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));									//get the properties of the first CUDA device

	float* gpu_gKernel;																				//pointer to the gaussian kerner
	unsigned char* gpu_inArray;  																	//pointer to input Array
	unsigned char* gpu_outArray_x;  																//pointer to output Array after convolution along x
	unsigned char* gpu_outArray_y;  																//pointer to output Array after all the convolution

	cout << "size: " << size << endl;

	HANDLE_ERROR(cudaMalloc(&gpu_gKernel, k_size * sizeof(float)));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&gpu_inArray, size * sizeof(char)));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&gpu_outArray_x, size_x * sizeof(char)));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&gpu_outArray_y, size_y * sizeof(char)));  							//allocate memory on device

	HANDLE_ERROR(cudaMemcpy(gpu_gKernel, gKernel, k_size * sizeof(float), cudaMemcpyHostToDevice));  //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(gpu_inArray, inArray, size * sizeof(char), cudaMemcpyHostToDevice));     //copy the array from main memory to device

	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	dim3 threads(blockDim, blockDim);
	cout << "threads.x: " << threads.x << endl;
	cout << "threads.y: " << threads.y << endl;
	dim3 blocks(width / threads.x + 1, height / threads.y + 1);


	// without shared memory
	cout << "--------------------- No shared memory ---------------------" << endl;
	cout << "Starts doing convolution on GPU without shared memory" << endl;

	//	utilize cudaEvent_t to serve as GPU timer
	cudaEvent_t d_start;
	cudaEvent_t d_stop;
	cudaEventCreate(&d_start);
	cudaEventCreate(&d_stop);
	cudaEventRecord(d_start, NULL);

	kernelConvolution_x_ns<<<blocks, threads>>>(gpu_outArray_x, gpu_inArray, gpu_gKernel, height, width, height_x, width_x, k_size);
	kernelConvolution_y_ns<<<blocks, threads>>>(gpu_outArray_y, gpu_outArray_x, gpu_gKernel, height_x, width_x, height_y, width_y, k_size);

	HANDLE_ERROR(cudaMemcpy(outArray_x, gpu_outArray_x, size_x * sizeof(char), cudaMemcpyDeviceToHost));  //copy the array back from device to main memory
	HANDLE_ERROR(cudaMemcpy(outArray_y, gpu_outArray_y, size_y * sizeof(char), cudaMemcpyDeviceToHost));  //copy the array back from device to main memory

	//	end of cudaEvent_t, calculate the time and show
	cudaEventRecord(d_stop, NULL);
	cudaEventSynchronize(d_stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, d_start, d_stop);
	cout << "It takes " << elapsedTime << " ms to do the GPU based convolution!" << endl;

	// output ppm file
 	char outFile_gpu_x_ns[50] = "./src/gpu_out_x_ns.ppm";
 	writePPM(outFile_gpu_x_ns, outArray_x, pSix, &width_x, &height_x, &maximum, &size_x);
 	char outFile_gpu_ns[50] = "./src/gpu_out_ns.ppm";
 	writePPM(outFile_gpu_ns, outArray_y, pSix, &width_y, &height_y, &maximum, &size_y);

 	cout << "Convolution on GPU without shared memory finished" << endl;

 	// using shared memory
	cout << "--------------------- Has shared memory ---------------------" << endl;
	cout << "Starts doing convolution on GPU with shared memory" << endl;

	//	utilize cudaEvent_t to serve as GPU timer
	cudaEventCreate(&d_start);
	cudaEventCreate(&d_stop);
	cudaEventRecord(d_start, NULL);

	// calculate the size of shared memory
	size_t sharedmem = 3 * blockDim * (blockDim + k_size - 1) * sizeof(char);
	cout << "sharedmem: " << sharedmem << endl;
	if(props.sharedMemPerBlock < sharedmem){
		std::cout<<"ERROR: insufficient shared memory"<<std::endl;
		exit(1);
	}
	kernelConvolution_x<<<blocks, threads, sharedmem>>>(gpu_outArray_x, gpu_inArray, gpu_gKernel, height, width, height_x, width_x, k_size);
	kernelConvolution_y<<<blocks, threads, sharedmem>>>(gpu_outArray_y, gpu_outArray_x, gpu_gKernel, height_x, width_x, height_y, width_y, k_size);

	HANDLE_ERROR(cudaMemcpy(outArray_x, gpu_outArray_x, size_x * sizeof(char), cudaMemcpyDeviceToHost));  //copy the array back from device to main memory
	HANDLE_ERROR(cudaMemcpy(outArray_y, gpu_outArray_y, size_y * sizeof(char), cudaMemcpyDeviceToHost));  //copy the array back from device to main memory

	//	end of cudaEvent_t, calculate the time and show
	cudaEventRecord(d_stop, NULL);
	cudaEventSynchronize(d_stop);
	cudaEventElapsedTime(&elapsedTime, d_start, d_stop);
	cout << "It takes " << elapsedTime << " ms to do the GPU based convolution!" << endl;

	// output ppm file
 	char outFile_gpu_x[50] = "./src/gpu_out_x.ppm";
 	writePPM(outFile_gpu_x, outArray_x, pSix, &width_x, &height_x, &maximum, &size_x);
 	char outFile_gpu[50] = "./src/gpu_out.ppm";
 	writePPM(outFile_gpu, outArray_y, pSix, &width_y, &height_y, &maximum, &size_y);

 	cout << "Convolution on GPU with shared memory finished" << endl;

 	// -------------------------------------- GPU VERSION ---------------------------------------- //

	return 0;
}


