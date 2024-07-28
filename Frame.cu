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
			int neighborIdx = i + 3 * (j * width + k);
			if (neighborIdx >= 0 && neighborIdx < width * height * 3)
				if (abs(in[neighborIdx] - in[i]) >= 50 ||
					abs(in[neighborIdx + 1] - in[i + 1]) >= 50 ||
					abs(in[neighborIdx + 2] - in[i + 2]) >= 50) 
						out[i / 3] = i / 3;
		}
	}
}
__global__ void findVerts(int* in, int* out, int size, int width, int height) {
	const int errorY = 2, errorX = 2;
	const int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x); // xCoord
	if (i >= size) return; // size is likely not divisor of gridsize
	if (i == 0) if (abs(in[i+1] - in[i+3]) > errorY) return; // drop isolated points if anyc
	if (i == size) 	if (abs(in[i-2] - in[i]) > errorY) return;
	else if (abs(in[i+1] - in[i+3]) > errorY || abs(in[i - 2] - in[i]) > errorY) return;
	// run binary search for point on X
	int left = 1, right = size - 1, mid;
	while (left <= right) {
		mid = (left + right) / 2; // xCoord 
		if (abs(in[mid] - in[i + 1]) < errorX) { out[i] = in[mid]; out[i + 1] = in[mid + 1]; }
		if (in[mid] < in[i + 1]) left = mid + 1;
		else right = mid - 1;
	}
}
void Frame::processFrame(Mat frame) {
	Vec3b data;
	int* inPL = 0;
	int* outPL = 0;

	cudaError_t inMalErrPL = cudaMallocManaged(&inPL, 3 * N * sizeof(int));
	if (inMalErrPL != cudaSuccess) { printf("Input Array Malloc Error: code %d - %s.\n", cudaError(inMalErrPL), cudaGetErrorString(inMalErrPL)); return; }
	cudaError_t outMalErrPL = cudaMalloc(&outPL, 3 * N * sizeof(int));
	if (outMalErrPL != cudaSuccess) { printf("Output Array Malloc Error: code %d - %s.\n", cudaError(outMalErrPL), cudaGetErrorString(outMalErrPL)); return; }
	cudaMemset(outPL, 0, 3 * N * sizeof(int));

	for (int i = 0; i < N; i += 3) {
		data = frame.at<Vec3b>(Point(i % 1080, i / 1080));
		for (int j = 0; j < 3; j++) 
			inPL[i + j] = static_cast<int>(data[j]);
	}
	
	findPoints<<<dimGrid, dimBlock>>>(inPL, outPL, frame.cols, frame.rows);

	int* hostOutPL = (int*)calloc(3 * N, sizeof(int));
	cudaMemcpy(hostOutPL, outPL, N * sizeof(int), cudaMemcpyDeviceToHost);

	cudaError_t syncErr = cudaGetLastError();
	cudaError_t asyncErr = cudaDeviceSynchronize();
	if (syncErr != cudaSuccess) { printf("Sync Kernel Error: code %d - %s.\n", cudaError(syncErr), cudaGetErrorString(syncErr)); throw invalid_argument("Sync Kernel Error"); }
	if (asyncErr != cudaSuccess) { printf("Async Kernel Error: code %d - %s.\n", cudaError(asyncErr), cudaGetErrorString(asyncErr)); throw invalid_argument("Sync Kernel Error"); }

	for (int i = 0; i < 3*N; i++) if (hostOutPL[i] != 0) pointList.push_back(hostOutPL[i]);
	
	cudaFree(inPL);
	cudaFree(outPL);

	int* inVL = 0;
	int* outVL = 0;

	cudaError_t inMalErrVL = cudaMallocManaged(&inVL, pointList.size());
	if (inMalErrVL != cudaSuccess) { printf("Input Array Malloc Error: code %d - %s.\n", cudaError(inMalErrVL), cudaGetErrorString(inMalErrVL)); return; }
	cudaError_t outMalErrVL = cudaMalloc(&outVL, pointList.size());
	if (outMalErrVL != cudaSuccess) { printf("Output Array Malloc Error: code %d - %s.\n", cudaError(outMalErrVL), cudaGetErrorString(outMalErrVL)); return; }
	cudaMemset(outVL, 0, pointList.size() * sizeof(int));

	for (int i = 0; i < pointList.size(); i++) inVL[i] = pointList.at(i);

	findVerts << <(pointList.size() + TPB - 1) / TPB, TPB>> > (inVL, outVL, pointList.size(), frame.cols, frame.rows);

	int* hostOutVL = (int*)calloc(pointList.size(), sizeof(int));
	cudaMemcpy(hostOutVL, outVL, pointList.size() * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < pointList.size(); i++) if (hostOutPL[i] != 0) vertList.push_back(hostOutVL[i]);

	cudaFree(inVL);
	cudaFree(outVL);
	
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
	for (int i = 0; i < vertList.size(); i += 2) {
		if (i != 0) cout << ", ";
		cout << "(";
		cout << vertList.at(i) % 1080 << ", " << vertList[i + 1] / 1080;
		cout << ")";
	}
	cout << "}\n----------------------------------------------------------------------------------------------\n";
}
