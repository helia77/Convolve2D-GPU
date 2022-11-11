//----Very first version of mine code.Some need to be modified for speed-up!
//----In this .cu file, I used the simple way to do GPU based convolution blur.
//----But __shared__ memory can also speed-up the process
//----Here is the __shared__ version:

//------------------------My programming assignment 1----------------------
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <streambuf>
#include <time.h>

using namespace std;

//Using syntax __syncthreads()
#pragma once
#ifdef __INTELLISENSE__
void __syncthreads();
#endif

//Using API call HADDLE_ERROR
static void HandleError( cudaError_t err, const char *file, int line ) {
if (err != cudaSuccess) {
cout<<cudaGetErrorString( err )<<" in"<< file <<" at line "<< line;
}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define pi	3.14159

//Input the variance for blurring
int sigma = 5;
int k = 6 * sigma + 1;

__global__ void convolution_x(unsigned char *I, unsigned char *O, float *KK, int k, unsigned int W, unsigned int w, unsigned int h){
	extern __shared__ unsigned char sharedPtr[];//(blockDim.x + k -1)* 3 * blockDim.y, remember row * col for 2d arrays
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iy = blockDim.y * blockIdx.y + threadIdx.y;
	int tidx = threadIdx.x;//-> col
	int tidy = threadIdx.y;//-> row
	if((ix) >= w || (iy) >= h)return;//Test for hanging threads to avoid segfaults
	int S = blockDim.x + k - 1;//How many pixels are need for one line, *blockDim.y equals to the number of pixels need for one block
	for(int si = tidx; si < S; si += blockDim.x){
		sharedPtr[3 * (tidy * S + si) + 0] = I[3 * (iy * W + blockDim.x * blockIdx.x + si) + 0];
		sharedPtr[3 * (tidy * S + si) + 1] = I[3 * (iy * W + blockDim.x * blockIdx.x + si) + 1];
		sharedPtr[3 * (tidy * S + si) + 2] = I[3 * (iy * W + blockDim.x * blockIdx.x + si) + 2];
	}
	__syncthreads();	

	float sumx[3] = {0};
	for(int ki = 0; ki < k; ki++){
		sumx[0] += KK[ki] * (float)sharedPtr[3 * (tidy * S + tidx + ki)+ 0];
		sumx[1] += KK[ki] * (float)sharedPtr[3 * (tidy * S + tidx + ki)+ 1];
		sumx[2] += KK[ki] * (float)sharedPtr[3 * (tidy * S + tidx + ki)+ 2];
	}
	O[3 * (iy * w + ix) + 0] = (unsigned char)sumx[0];
	O[3 * (iy * w + ix) + 1] = (unsigned char)sumx[1];
	O[3 * (iy * w + ix) + 2] = (unsigned char)sumx[2];

}

__global__ void convolution_y(unsigned char *I, unsigned char *O, float *KK, int k, unsigned int W, unsigned int w, unsigned int h){
	extern __shared__ unsigned char sharedPtr[];//(blockDim.y + k -1)* 3 * blockDim.x
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iy = blockDim.y * blockIdx.y + threadIdx.y;
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;
	if(ix >= w || iy >= h)return;//Test for hanging threads to avoid segfaults
	int S = blockDim.y + k - 1;//How many pixels are need for one line, *blockDim.x equals to the number of pixels need for one block
	for(int si = tidy; si < S; si += blockDim.y){
		sharedPtr[3 * (tidx * S + si) + 0] = I[3 * ((blockDim.y * blockIdx.y + si) * W + ix) + 0];
		sharedPtr[3 * (tidx * S + si) + 1] = I[3 * ((blockDim.y * blockIdx.y + si) * W + ix) + 1];
		sharedPtr[3 * (tidx * S + si) + 2] = I[3 * ((blockDim.y * blockIdx.y + si) * W + ix) + 2];
	}
	__syncthreads();	

	float sumy[3] = {0};
	for(int ki = 0; ki < k; ki++){
		sumy[0] += KK[ki] * (float)sharedPtr[3 * (tidy + ki + tidx * S) + 0];
		sumy[1] += KK[ki] * (float)sharedPtr[3 * (tidy + ki + tidx * S) + 1];
		sumy[2] += KK[ki] * (float)sharedPtr[3 * (tidy + ki + tidx * S) + 2];
	}
	O[3 * (iy * w + ix) + 0] = (unsigned char)sumy[0];
	O[3 * (iy * w + ix) + 1] = (unsigned char)sumy[1];
	O[3 * (iy * w + ix) + 2] = (unsigned char)sumy[2];
}

//!!!!!!!!!!!!!!!!!The next time remember it is not necessary to copy one single value from host to device!!!!!!!!!!!!!!!!!!!!!!
//Kernel for convolution along x axis
//__global__ void convolution_x(unsigned char *I, unsigned char *O, float *KK, int kk, unsigned int W, unsigned int w, unsigned int h){
//	unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
//	unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
//	//Robust way is to throw away the edge pixel
//	if(row >= h || col >= w)return;
//	float sumx[3] = {0};//Every thread has one array of 3 elements stored in register
//	for(int j = 0; j < 3; j++){//For three channels
//		for(int i = 0; i < kk; i++){//Every thread do a product with elements from the Gaussian kernel in one time
//			sumx[j] += KK[i] * (float)I[3 * (row * W + col + i) + j];
//		}
//		__syncthreads();		//It is not necessary to do synchronization
//		O[3 * (row * w + col) + j] = (unsigned char)sumx[j];//Write back data in terms of channel
//	}
//}

//Kernel for convolution along y axis
//__global__ void convolution_y(unsigned char *I, unsigned char *O, float *KK, int kk, unsigned int W, unsigned int w, unsigned int h){
//	unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
//	unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
//	//Everything is the same as convolution along x axis but be careful about the threads actions
//	if(row >= h || col >= w)return;
//	float sumy[3] = {0};
//	for(int j = 0; j < 3; j++){
//		for(int i = 0; i < kk; i++){
//			sumy[j] += KK[i] * (float)I[3 * ((row + i) * W + col) + j];//Not coalescing, but it might be Ok.
//		}
//		__syncthreads();//It is not necessary to do synchronization
//		O[3 * (row * w + col) + j] = (unsigned char)sumy[j];
//	}
//}


int main(){

	//Load the image into vector
	ifstream ifile;
	ifile.open("whirlpool.ppm", ios::binary);
	if(!ifile){
		cout<<"open error!"<<endl;
	}

	//Variables for image information
	unsigned int width_x;
	unsigned int height_x;
	unsigned int width_y;
	unsigned int height_y;
	unsigned int Width, Height, MaximumValue;
	string FileType;

	//Get the filetype of the PPM image
	getline(ifile, FileType, '\n');
	if( FileType != "P6")
		cout<<"Invalid PPM header or signature!"<<endl;
	
	string ss;//String for transfer data

	//Skip comments
	while(getline(ifile, ss)){
		if(ss[0] == '#')
			continue;
		else
			break;
	}	

	//Read image size
	istringstream iss;  
	iss.str(ss);
	iss>>Width>>Height;
	int pixel = Width * Height;
	if ( Width <= 0 || Height <= 0 )  
		cout<<"Invalid image size!"<<endl;  

	//Find the maximum value in the PPM file
	if (getline(ifile, ss))  {  
		iss.clear();  
		iss.str(ss);  
		iss>>MaximumValue; 
	}

	//Load the rest of image into a string, be careful that these are chars
	string rgb( ( istreambuf_iterator<char>(ifile) ),istreambuf_iterator<char>() );   

	//Close the file
	ifile.close();

	//Show the basic information you got
	cout<<"FileType = "<<FileType<<endl;
	cout<<"Width = "<<Width<<endl;
	cout<<"Height = "<<Height<<endl;
	cout<<"MaximumValue = "<<MaximumValue<<endl;
	cout<<"Number of pixel = "<<pixel<<endl;

	//Create Gaussian filter
	float* K = new float[k];
	for(int i = 0; i < k; i++){
		//Utilize the transform
		float x = (float)i - (float)(k/2);
		float g = (float)1.0 / (sigma * sqrt(2 * (float)pi)) * exp( - (x * x) / (2 * sigma * sigma) );
		K[i] = g;
	}

	//Create timer
	//clock_t start, finish;
	//double duration;
	//start = clock();

	//Remember to expand the size for we have pixels that are larger than 128 into get the original image pixels
	//It can be simply forced to transfer!!!
	unsigned char *Trgb = new unsigned char[Width * Height * 3];
	for(unsigned int i = 0; i < Width * Height; i++){
		if(rgb[3 * i] < 0)
			Trgb[3 * i] = rgb[3 * i] + 256;
		else
			Trgb[3 * i] = rgb[3 * i];
		if(rgb[3 * i + 1] < 0)
			Trgb[3 * i + 1] = rgb[3 * i + 1] + 256;
		else
			Trgb[3 * i + 1] = rgb[3 * i + 1];
		if(rgb[3 * i + 2] < 0)
			Trgb[3 * i + 2] = rgb[3 * i + 2] + 256;
		else
			Trgb[3 * i + 2] = rgb[3 * i + 2];
	}

	//-------------------To begin with CPU parts----------------------
	//Do convolution along x - direction
	//float *sumx = new float[3];
	//sumx[0] = sumx[1] = sumx[2] = 0.0f;
	width_x = Width - (k/2)*2;
	height_x = Height;
	//unsigned char *rgb_x = new unsigned char[width_x * height_x * 3];
	//
	//for(unsigned int x = 0; x < width_x; x++){
	//	for(unsigned int y = 0; y < height_x; y++){
	//		for(int i = 0; i < k; i++){
	//			sumx[0] += K[i] * (float)Trgb[3 * (y * Width + x + i)];//Cope with it in terms of three channels
	//			sumx[1] += K[i] * (float)Trgb[3 * (y * Width + x + i) + 1];
	//			sumx[2] += K[i] * (float)Trgb[3 * (y * Width + x + i) + 2];
	//		}
	//		//Make sure that it is write in order
	//		rgb_x[3 * (y * width_x + x)] = (unsigned char)sumx[0];
	//		rgb_x[3 * (y * width_x + x) + 1] = (unsigned char)sumx[1];
	//		rgb_x[3 * (y * width_x + x) + 2] = (unsigned char)sumx[2];
	//		sumx[0] = 0.0f;//Don't forget to initialize the iterative sums
	//		sumx[1] = 0.0f;
	//		sumx[2] = 0.0f;
	//	}
	//}
	//
	////Do convolution along y - direction
	//float *sumy = new float[3];
	//sumy[0] = sumy[1] = sumy[2] = 0.0f;
	width_y = Width - (k/2)*2;
	height_y = Height - (k/2)*2;
	//unsigned char *rgb_y = new unsigned char[width_y * height_y * 3];
	//
	////My code is a little bit different from what you taught on class. You are move the y-window from left to right, then up to down. But mine is move up to down, then left to right.
	//for(unsigned int y = 0; y < height_y; y++){
	//	for(unsigned int x = 0; x < width_y; x++){
	//		for(int i = 0; i < k; i++){
	//			sumy[0] += K[i] * (float)rgb_x[3* ((y + i) * width_x + x)];
	//			sumy[1] += K[i] * (float)rgb_x[3* ((y + i) * width_x + x) + 1];
	//			sumy[2] += K[i] * (float)rgb_x[3* ((y + i) * width_x + x) + 2];
	//		}
	//		//Make sure that it is write in order
	//		rgb_y[3 * (y * width_y + x)] = (unsigned char)sumy[0];
	//		rgb_y[3 * (y * width_y + x) + 1] = (unsigned char)sumy[1];
	//		rgb_y[3 * (y * width_y + x) + 2] = (unsigned char)sumy[2];
	//		sumy[0] = 0.0f;
	//		sumy[1] = 0.0f;
	//		sumy[2] = 0.0f;
	//	}
	//}
	//
	////End of time for the computation
	//finish = clock();
	//duration = (double)(finish - start)/CLOCKS_PER_SEC;
	//cout<<"It takes "<<duration<<" s to do the CPU based convolution!"<<endl;
	//
	////Create a new empty file for store the output PPM image
	//ofstream ofp;
	//ofp.open("CPU_Cutee.ppm", ios::binary);
	//
	////Output the head file
	//ofp<<"P6"<<endl<<width_y<<" "<<height_y<<endl<<255<<endl;
	//
	////Tring to make it faster!!
	//for(unsigned int i = 0; i < width_y * height_y; i++){
	//	ofp<<rgb_y[3 * i]<<rgb_y[3 * i + 1]<<rgb_y[3 * i + 2];
	//}
	//
	////Close the output file
	//ofp.close();
	//
	////Always remember to delete dynamic allocations
	//delete[] sumx;
	//delete[] sumy;
	//delete[] rgb_x;
	//delete[] rgb_y;

	//----------------------From now on, starts the GPU parts!----------------------------

	unsigned char *h_rgb = new unsigned char[width_y * height_y * 3];

	//Declare the variables that are need in device
	//Single values do not need copy from host to device! Waste of time
	float *d_K;
	unsigned char *d_rgb;
	unsigned char *d_rgb_x;
	unsigned char *d_rgb_y;


	//Allocate memory for those variables. Always check errors!!!!!!!!!!!!!!!!!!!!
	HANDLE_ERROR (cudaMalloc(&d_rgb, Width * Height * 3 * sizeof(char)));
	HANDLE_ERROR (cudaMalloc(&d_rgb_x, width_x * height_x * 3 * sizeof(char)));
	HANDLE_ERROR (cudaMalloc(&d_K, k * sizeof(float)));
	HANDLE_ERROR (cudaMalloc(&d_rgb_y, width_y * height_y * 3 * sizeof(char)));

	//Copy necessary data from host to device
	HANDLE_ERROR (cudaMemcpy(d_rgb, Trgb, Width * Height * 3 * sizeof(char), cudaMemcpyHostToDevice));
	HANDLE_ERROR (cudaMemcpy(d_K, K, k * sizeof(float), cudaMemcpyHostToDevice));

	//Utilize cudaEvent_t to serve as GPU timer
	//cudaEvent_t d_start;
	//cudaEvent_t d_stop;
	//cudaEventCreate(&d_start);
	//cudaEventCreate(&d_stop);
	//cudaEventRecord(d_start, NULL);

	//Run the kernels on device
	dim3 blocksx(width_x / 32 + 1, height_x / 32 + 1);//using square blocks, two dimensional grid
	dim3 threadsx(32, 32);
	size_t sharedmem = 3 * 32 * (32 + k - 1) * sizeof(unsigned char);

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);
	if(props.sharedMemPerBlock < sharedmem){
		std::cout<<"ERROR: insufficient shared memory"<<std::endl;
		exit(1);
	}

	convolution_x<<<blocksx, threadsx, sharedmem>>>(d_rgb, d_rgb_x, d_K, k, Width, width_x, height_x);
	//convolution_x<<<dim3(width_x/1024+1,height_x), 1024>>>(d_rgb, d_rgb_x, d_K, d_k, d_Width, d_width_x, d_height_x);

	dim3 blocksy(width_y / 32 + 1, height_y / 32 + 1);
	dim3 threadsy(32, 32);
	convolution_y<<<blocksy, threadsy, sharedmem>>>(d_rgb_x, d_rgb_y, d_K, k, width_x, width_y, height_y);
	//convolution_y<<<dim3(width_x,height_x/1024+1), dim3(1,1024)>>>(d_rgb_x, d_rgb_y, d_K, d_k, d_width_x, d_width_y, d_height_y);

	//End of cudaEvent_t, calculate the time and show
	//cudaEventRecord(d_stop, NULL);
	//cudaEventSynchronize(d_stop);
	//float elapsedTime;
	//cudaEventElapsedTime(&elapsedTime, d_start, d_stop);
	//cout<<"It takes "<<elapsedTime<<" ms to do the GPU based convolution!"<<endl;

	//Copy data back to Host, the very first time I made a typo that using "width_y * width_y" and you know it is hard to detect this kind of error without HANDLE_ERROR
	HANDLE_ERROR (cudaMemcpy(h_rgb, d_rgb_y, width_y * height_y * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));

	//Output the file
	ofstream of;
	of.open("GPU_Cutee.ppm", ios::binary);

	//Output the head file
	of<<"P6"<<endl<<width_y<<" "<<height_y<<endl<<255<<endl;

	//Tring to make it faster!!
	for(unsigned int i = 0; i < width_y * height_y; i++){
		of<<h_rgb[3 * i]<<h_rgb[3 * i + 1]<<h_rgb[3 * i + 2];
	}

	//Close the output file
	of.close();

	//Remember to delete dynamic allocation memorys
	delete[] K;
	delete[] Trgb;
	delete[] h_rgb;

	//Reset the device, also see in cudaDeviceReset
	cudaFree(d_K);
	cudaFree(d_rgb);
	cudaFree(d_rgb_x);
	cudaFree(d_rgb_y);


	return 0;
}