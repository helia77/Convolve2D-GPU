#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>

using namespace std;


#define PI 3.141593

static void HandleError( cudaError_t err, const char *file, int line ) {
	if (err != cudaSuccess) {
		cout<<cudaGetErrorString( err )<<"in"<< file <<"at line"<< line;
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


__global__ void kernelMatrixMult(float* C, float* A, float* B, size_t M, size_t N){
	size_t i = blockIdx.y * blockDim.y + threadIdx.y;  //calculate the i (row) index
	size_t j = blockIdx.x * blockDim.x + threadIdx.x;  //calculate the j (column) index
	if(i >= M || j >= N) return;  //return if (i,j) is outside the matrix
	float c = 0;  //initialize a register to store the result
	for(size_t n = 0; n < M; n++)  //for each element in the dot product
		c += A[i*M+n] * B[n*N+j];  //perform a multiply-add
	C[i*N + j] = c;  //send the register value to global memory
}


void convolution(float* C, float* A, float* B, size_t M, size_t N, size_t K){
	float c[3] = 0;
	size_t length = (N - K) + 1
	for (int i = 0; i < length; i++){
		for (int k)
	}

	float c = 0;
	for (int i = K / 2; i < M - K / 2; i++){
		for (int j = K / 2; j < N - K / 2; j++){
			for (int u = - K / 2; u < K / 2; u++){
				for (int v = - K / 2; v < K / 2; v++){
					c += B[i * K + j] * A[(i - u) * N + (j - v)]
				}
			}
			C[i * (N - K) + j] = c
		}
	}

}

void MatrixMult(float* C, float* A, float* B, size_t M, size_t N){
	for(int i = 0; i < M; i++){
		for(int j = 0; j < N; j++){
			float c = 0;  //initialize a register to store the result
			for(size_t n = 0; n < M; n++) { //for each element in the dot product
				c += A[i*M+n] * B[n*N+j];  //perform a multiply-add
			}
			C[i*N + j] = c;  //send the register value to global memory
		}
	}
}

unsigned char* readPPM(const char* fileName, char* pSix, int* width, int* height, int* maximum) {

	// open the file to read just the header reading
	FILE* fr = fopen(fileName, "r");

	// formatted read of header
	fscanf(fr, "%s", pSix);
	cout << "pSix: " << pSix << endl;

	// check to see if it's a PPM image file
	if (strncmp(pSix, "P6" , 10) != 0) {
		printf("They are not the same\n");
	} else {
		printf("They are the same\n");
	}

	// read the rest of header
	fscanf(fr, "%d\n %d\n", width, height);

    fscanf(fr, "%d\n", maximum);

    cout << "pSix: " << pSix << endl;
    cout << "Width: " << width << endl;
    cout << "Height: " << height << endl;


    // check to see if they were stored properly
    printf("PSix: %s\n", pSix);
    printf("Width: %d\n", *width);
    printf("Height: %d\n", *height);
    printf("maximum: %d\n", *maximum);

    int size = *width * *height;
    // int size = 423800;

    // allocate array for pixels
    unsigned char* pixels = new unsigned char[size];
//    unsigned char* pixels = (unsigned char*) malloc(size * sizeof(char));


	// unformatted read of binary pixel data
    fread(pixels, sizeof(char), size, fr);
    printf("%d\n", *pixels);
    pixels++;
    printf("%d\n", *pixels);
    pixels++;
    printf("%d\n", *pixels);
    printf("%d\n", (float)pixels[0]);
    printf("%d\n", (float)pixels[1]);
    printf("%d\n", (float)pixels[2]);

//	while (fread(pixels, sizeof(int), 128, fr)) {
//		printf("%s", pixels);
//	} // end of for loop

	// close file
	fclose(fr);

	// return the array
	return pixels;

} // end of readPPM

//istream& operator >>(istream &inputStream, PPMObject &other)
//{
//    inputStream >> other.magicNum;
//    inputStream >> other.width >> other.height >> other.maxColVal;
//    inputStream.get(); // skip the trailing white space
//    size_t size = other.width * other.height * 3;
//    other.m_Ptr = new char[size];
//    inputStream.read(other.m_Ptr, size);
//    return inputStream;
//}



int main(int argc, char* argv[]){							//main function											//GPU console output


	float miu = 0;
	float sigma = 40;
	int k_size = 21;

//	char fileName[50] = "Galaxy.ppm";
	char fileName[50] = "./src/Aerial.512.ppm";
	char pSix[10];		// indicates this is a PPM image
	int width = 0;		// width of the image
	int height = 0;		// height of the image
	int maximum = 0;	// maximum pixel value
	int size = 0;		// size of the array
//
	cout << "read starts" << endl;
	// read the PPM file and store its contents inside an array and return the pointer to that array to pixelArray
	unsigned char* inArray = readPPM(fileName, pSix, &width, &height, &maximum);


 	float* gKernal = (float*) malloc(k_size * sizeof(float));
 	for (int i = 0, i < k_size, i++){
 		gKernal[i] = 1 / sqrt(2 * sigma * sigma * PI) * exp(- pow((i - miu), 2) / (2 * sigma * sigma));;
 	}

 	float* outArray = (float*) malloc(size * sizeof(float));

 	convolution(outArray, inArray, gKernal, height, width)



	ifstream infile(fileName, ios::binary);

	string line;
	getline(infile, line);
	if (line == "P6"){
		cout << "correct fileType!" << endl;
	}
	cout << "line1" << line << endl;
	getline(infile, line);


	cout << "line2" << line << endl;


//	char* p = (char*) malloc(2 * sizeof(char));
//	infile.read(p, 2 * sizeof(char));
//	cout << "M" << p << endl;


	cout << "function return" << endl;
//	cout << pixelArray << endl;
	cout << "read finished" << endl;



//
//	char data[100];





//	infile >> data;
//	cout << data << endl;

//	float* A;
//	float* B;
//	float* C;
//	float a[2][2] = {{1, 2}, {3, 4}};
//	float b[2][3] = {{0, 1, 2}, {3, 4, 5}};
//	float c[2][3];
//	A = a;
//	B = b;
//	C = c;

// ------------------------------------------------------------------------------------------- //
//	float a[4] = {1, 2, 3, 4};								//A[2][2]
//	float b[6] = {0, 1, 2, 3, 4, 5};						//B[2][3]
//
//	size_t M, N;  //matrix sizes
//
//	M = 2;
//	N = 3;
//
//	float* A;
//	float* B;
//	float* C;
//	A = (float*) malloc(M * M * sizeof(float));			//allocate an array in main memory
//	B = (float*) malloc(N * M * sizeof(float));			//allocate an array in main memory
//	C = (float*) malloc(N * M * sizeof(float)); 		//allocate an array in main memory
//
//	A = a;
//	B = b;
//
//	cout << "matrix multiplication" << endl;
//
//	// ------------------ CPU version ------------------ //
//	MatrixMult(C, A, B, M, N);
//	// ------------------ CPU version ------------------ //
//
////	// ------------------ GPU version ------------------ //
////
////	cudaDeviceProp props;																//declare a CUDA properties structure
////	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));									//get the properties of the first CUDA device
////
////	float* gpu_A;  																		//pointer to A, A i,j = A[i*M+j]
////	float* gpu_B;  																		//pointer to B, B i,j = B[i*N+j]
////	float* gpu_C;  																		//pointer to C, C i,j = C[i*N+j]
////	HANDLE_ERROR(cudaMalloc(&gpu_A, M * M * sizeof(float)));  							//allocate memory on device
////	HANDLE_ERROR(cudaMalloc(&gpu_B, N * M * sizeof(float)));  							//allocate memory on device
////	HANDLE_ERROR(cudaMalloc(&gpu_C, N * M * sizeof(float)));  							//allocate memory on device
////
////	HANDLE_ERROR(cudaMemcpy(gpu_A, A, M * M * sizeof(float), cudaMemcpyHostToDevice));  //copy the array from main memory to device
////	HANDLE_ERROR(cudaMemcpy(gpu_B, B, N * M * sizeof(float), cudaMemcpyHostToDevice));  //copy the array from main memory to device
////
////	dim3 threads(sqrt(props.maxThreadsPerBlock), sqrt(props.maxThreadsPerBlock));
////	cout << "threads.x: " << threads.x << endl;
////	cout << "threads.y: " << threads.y << endl;
////	dim3 blocks(N/threads.x+1, M/threads.y+1);
////	kernelMatrixMult<<<blocks, threads>>>(gpu_C, gpu_A, gpu_B, M, N);
////
////	HANDLE_ERROR(cudaMemcpy(C, gpu_C, N * M * sizeof(float), cudaMemcpyDeviceToHost));  //copy the array back to main memory
////	// ------------------ GPU version ------------------ //
//
//	for(int i = 0; i < M; i++){
//		for(int j = 0; j < N; j++){
//			cout << "c[" << i << "][" << j << "]:";
//			cout << C[i*N + j] << endl;
//		}
//	}
//

// ------------------------------------------------------------------------------------------- //
}


