#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <cmath>
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
	processFrame(frame);
}
__global__ void findPoints(int* in, int* out, int width, int height) {
	const int i = 3 * (blockIdx.x * blockDim.x + threadIdx.x);
	int xCoord = (i / 3) % width;
	int yCoord = (i / 3) / width;
	if (xCoord == 0 || xCoord == width - 1 || yCoord == 0 || yCoord == height - 1) return;

	for (int j = -1; j <= 1; j++) {
		for (int k = -1; k <= 1; k++) {
			if (j == 0 && k == 0) continue;
			int neighborIdx = i + 3 * (j * width + k); // Calculate the neighbor index
			if (neighborIdx >= 0 && neighborIdx < width * height * 3)
				if (abs(in[neighborIdx] - in[i]) >= 50 ||
					abs(in[neighborIdx + 1] - in[i + 1]) >= 50 ||
					abs(in[neighborIdx + 2] - in[i + 2]) >= 50) 
						out[i / 3] = i / 3;
		}
	}
}
void Frame::processFrame(Mat frame) {
	Vec3b data;
	int* in = 0;
	int* out = 0;

	cudaError_t inMalErr = cudaMallocManaged(&in, 3 * N * sizeof(int));
	if (inMalErr != cudaSuccess) { printf("Input Array Malloc Error: code %d - %s.\n", cudaError(inMalErr), cudaGetErrorString(inMalErr)); return; }
	cudaError_t outMalErr = cudaMalloc(&out, 3 * N * sizeof(int));
	if (outMalErr != cudaSuccess) { printf("Output Array Malloc Error: code %d - %s.\n", cudaError(outMalErr), cudaGetErrorString(outMalErr)); return; }

	for (int i = 0; i < N; i += 3) {
		data = frame.at<Vec3b>(Point(i % 1080, i / 1080));
		for (int j = 0; j < 3; j++) 
			in[i + j] = static_cast<int>(data[j]);
	}
	
	findPoints<<<dimGrid, dimBlock>>>(in, out, frame.cols, frame.rows);

	int* hostOut = (int*)calloc(3 * N, sizeof(int));
	cudaMemcpy(hostOut, out, N * sizeof(int), cudaMemcpyDeviceToHost);

	cudaError_t syncErr = cudaGetLastError();
	cudaError_t asyncErr = cudaDeviceSynchronize();
	if (syncErr != cudaSuccess) { printf("Sync Kernel Error: code %d - %s.\n", cudaError(syncErr), cudaGetErrorString(syncErr)); throw invalid_argument("Sync Kernel Error"); }
	if (asyncErr != cudaSuccess) { printf("Async Kernel Error: code %d - %s.\n", cudaError(asyncErr), cudaGetErrorString(asyncErr)); throw invalid_argument("Sync Kernel Error"); }

	for (int i = 0; i < 3*N; i++) {
		if (hostOut[i] != 0) pointList.push_back(hostOut[i]);
	}

	cudaFree(in);
	cudaFree(out);

	printLists();
}
void Frame::printLists() {
	cout << "----------------------------------------------------------------------------------------------\n";
	cout << "POINT LIST:\n{ ";
	for (int i = 0; i < pointList.size(); i+=2) {
		if (i != 0) cout << ", ";
		cout << "(";
		cout << pointList.at(i)%1080 << ", " << pointList[i + 1]/1080;
		cout << ")";
	}
	cout << "}\n----------------------------------------------------------------------------------------------\n";
	cout << "VERTICE LIST:\n{ ";
	for (int i = 0; i < vertList.size(); i++) {
		if (i != 0) cout << ", ";
		cout << "(";
		for (int j = 0; j < vertList[i].size(); j++) {
			cout << vertList[i][j];
			if (j < vertList[i].size() - 1) cout << ", ";
		}
		cout << ")";
	}
	cout << "}\n----------------------------------------------------------------------------------------------\n";
}
