#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <vector>
#include "Frame.cuh"
#include "Video.cuh"
#include "processingFunc.cuh"

using namespace cv;
using namespace std;

#define N (1920*1080)
#define TPB 1024
const int GRIDSIZE = (N + TPB - 1) / TPB;
dim3 dimGrid = dim3(GRIDSIZE);
dim3 dimBlock = dim3(TPB);

Frame::Frame(Mat frame) {
	Vec3b* data = &frame.at<Vec3b>(Point(0, 0));
	processFrame(data);
}
__global__ void findPoints(int* in, thrust::device_ptr<int>out) {
	const int i = 3 * (blockIdx.x * blockDim.x + threadIdx.x);
	int xCoord = i / 1080;
	int yCoord = i % 1080;
	
	for (int j = -1081; j <= 1081; j++) {
		if (j == -1078) { j = -1; continue; }
		if (j == 2) { j = 1079; continue; }
		if (abs(in[i + j] - in[i]) < 50) {
			if (abs(in[i + j + 1] - in[i]) < 50)
				if (abs(in[i + j + 2] - in[i]) < 50)
					continue;
		}
		else out[i] = i / 3;
	}
}
void Frame::processFrame(Vec3b* dataPtr) {
	Vec3b data = *dataPtr;
	int* in = 0;
	thrust::host_vector<int> out(3 * N);
	thrust::device_vector<int> devVec(3*N);
	void* raw;

	cudaError_t inMalErr = cudaMallocManaged(&in, 3 * N * sizeof(int));
	if (inMalErr != cudaSuccess)  printf("Input Array Malloc Error: code %d - %s.\n", cudaError(inMalErr), cudaGetErrorString(inMalErr)); return; 
	cudaError_t outMalErr = cudaMalloc(&raw, 3 * N * sizeof(int));
	if (outMalErr != cudaSuccess) printf("Output Array Malloc Error: code %d - %s.\n", cudaError(outMalErr), cudaGetErrorString(outMalErr)); return;

	thrust::device_ptr<int> vecPtr = thrust::device_pointer_cast(raw);

	for (int i = 0; i < N; i += 3)
		for (int j = 0; j < 3; j++)
			in[i + j] = static_cast<int>(data[j]);
	
	findPoints<<<dimGrid, dimBlock>>>(in, vecPtr);

	cudaError_t syncErr = cudaGetLastError();
	cudaError_t asyncErr = cudaDeviceSynchronize();
	if (syncErr != cudaSuccess) { printf("Sync Kernel Error: code %d - %s.\n", cudaError(syncErr), cudaGetErrorString(syncErr)); throw invalid_argument("Sync Kernel Error"); }
	if (asyncErr != cudaSuccess) { printf("Async Kernel Error: code %d - %s.\n", cudaError(asyncErr), cudaGetErrorString(asyncErr)); throw invalid_argument("Sync Kernel Error"); }

	thrust::copy(devVec.begin(), devVec.end(), out.begin());

	cudaFree(in);
	cudaFree(raw);
}
void Frame::printLists() {
	cout << "----------------------------------------------------------------------------------------------\n";
	cout << "{ ";
	for (int i = 0; i < pointList.size(); i++) {
		if (i != 0) cout << ", ";
		cout << "(";
		for (int j = 0; j < pointList[i].size(); j++) {
			cout << pointList[i][j];
			if (j < pointList[i].size() - 1) cout << ", ";
		}
		cout << ")";
	}
	cout << " }\n----------------------------------------------------------------------------------------------\n";
	cout << "{ ";
	for (int i = 0; i < vertList.size(); i++) {
		if (i != 0) cout << ", ";
		cout << "(";
		for (int j = 0; j < vertList[i].size(); j++) {
			cout << vertList[i][j];
			if (j < vertList[i].size() - 1) cout << ", ";
		}
		cout << ")";
	}
	cout << "----------------------------------------------------------------------------------------------\n";
}
