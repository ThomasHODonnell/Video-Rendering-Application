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

const int N = (1920 * 1080), TPB = 1024;
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
					out[(i / 3) * 2] = 1; // 0 - N-1 indexed, return more information? 
				else out[(i / 3) * 2] = 0; 
		}
	}
}
__device__ int binarySearch(int* in, int left, int right, int errorX, int errorY, int refX, int refY) {
	if (right >= left) {
		int mid = (left + right / 2);
		if (mid % 2 == 0 && mid != right) mid++; // yCoord
		if (mid > right || mid - 1 < left) return -1;
		if (abs(in[mid] - refY) < errorY && abs(in[mid - 1] - refX) < errorX) return mid;
		if (in[mid] < refY || (in[mid] == refY && in[mid - 1] < refX)) return binarySearch(in, mid + 1, right, errorX, errorY, refX, refY);
		else return binarySearch(in, left, mid - 1, errorX, errorY, refX, refY);
	}
	return -1;
}

__global__ void findVerts(int* in, int* out, int size) {
	const int errorY = 5, errorX = 5;
	const int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x); // xCoord
	if (i >= size) return;
	if (i == 0) return;
		//if (abs(in[i + 1] - in[i + 3]) > errorY) return; // drop isolated points if any
	else if (i == size - 2) return;
		//if (abs(in[i-1] - in[i+1]) > errorY) return;
	else if (abs(in[i+1] - in[i+3]) > errorY || abs(in[i-1] - in[i+1]) > errorY) return;

	int mid = binarySearch(in, 0, size - 1, errorX, errorY, in[i], in[i + 1]);
	if (mid != -1) {
		float slope1, slope2, slope3; 
		if (in[i] - in[mid-1] != 0) slope1 = FLT_MAX;
		else slope1 = abs(static_cast<float>(in[i + 1] - in[mid])) / static_cast<float>(abs(in[i] - in[mid - 1]));
		if (in[i] - in[i-2] != 0) slope2 = FLT_MAX;
		else slope2 = abs(static_cast<float>(in[i + 1] - in[i - 1])) / static_cast<float>(abs(in[i] - in[i - 2]));
		if (in[i] - in[i+2] != 0) slope3 = FLT_MAX;
		else slope3 = abs(static_cast<float>(in[i + 1] - in[i + 3])) / static_cast<float>(abs(in[i] - in[i + 2]));

		if (slope1 != slope2 || slope1 != slope3) {
			out[i] = in[mid - 1];
			out[i + 1] = in[mid];
		}
		else out[i] = out[i + 1] = 0; 
	}
}
void Frame::processFrame(Mat frame) {
	Vec3b data;
	int* inPL = 0;
	int* outPL = 0; 

	cudaError_t inMalErrPL = cudaMallocManaged(&inPL, 3 * N * sizeof(int)); // RGB * N
	if (inMalErrPL != cudaSuccess) { printf("Input Array Malloc Error: code %d - %s.\n", cudaError(inMalErrPL), cudaGetErrorString(inMalErrPL)); return; }
	cudaError_t outMalErrPL = cudaMalloc(&outPL, 2 * N * sizeof(int));
	if (outMalErrPL != cudaSuccess) { printf("Output Array Malloc Error: code %d - %s.\n", cudaError(outMalErrPL), cudaGetErrorString(outMalErrPL)); return; }

	for (int i = 0; i < N; i += 3) { // can this be done faster??
		data = frame.at<Vec3b>(Point(i % 1080, i / 1080));
		for (int j = 0; j < 3; j++) 
			inPL[i + j] = static_cast<int>(data[j]);
	}
	
	findPoints<<<dimGrid, dimBlock>>>(inPL, outPL, frame.cols, frame.rows);

	pointList = (int*)malloc(2 * N * sizeof(int)); // X,Y * N
	cudaMemcpy(pointList, outPL, 2 * N * sizeof(int), cudaMemcpyDeviceToHost);

	cudaError_t syncErrPL = cudaGetLastError();
	cudaError_t asyncErrPL = cudaDeviceSynchronize();
	if (syncErrPL != cudaSuccess) { printf("Sync Kernel Error: code %d - %s.\n", cudaError(syncErrPL), cudaGetErrorString(syncErrPL)); throw invalid_argument("Sync Kernel Error"); }
	if (asyncErrPL != cudaSuccess) { printf("Async Kernel Error: code %d - %s.\n", cudaError(asyncErrPL), cudaGetErrorString(asyncErrPL)); throw invalid_argument("Async Kernel Error"); }

	cudaFree(inPL);
	cudaFree(outPL);

	int* outVL = 0;

	cudaError_t outMalErrVL = cudaMalloc(&outVL, 2 * N * sizeof(int));
	if (outMalErrVL != cudaSuccess) { printf("Output Array Malloc Error: code %d - %s.\n", cudaError(outMalErrVL), cudaGetErrorString(outMalErrVL)); return; }

	findVerts << <(2 * N + 32 - 1) / 32, 32 >> > (pointList, outVL, 2 * N);

	vertList = (int*)malloc(2 * N * sizeof(int));
	cudaMemcpy(vertList, outVL, 2 * N * sizeof(int), cudaMemcpyDeviceToHost);

	cudaError_t syncErrVL = cudaGetLastError();
	cudaError_t asyncErrVL = cudaDeviceSynchronize();
 	if (syncErrVL != cudaSuccess) { printf("Sync Kernel Error: code %d - %s.\n", cudaError(syncErrVL), cudaGetErrorString(syncErrVL)); throw invalid_argument("Sync Kernel Error"); }
	if (asyncErrVL != cudaSuccess) { printf("Async Kernel Error: code %d - %s.\n", cudaError(asyncErrVL), cudaGetErrorString(asyncErrVL)); throw invalid_argument("Async Kernel Error"); }

	Vec3b yel = { 255, 255, 0 };
	for (int i = 0; i < 2 * N; i++) 
		if (vertList[i] != 0) 
			frame.at<Vec3b>(Point(i % 1080, i / 1080)) = yel;
	
	imshow("Vert List Output", frame);

	cudaFree(outVL);
	
	printLists(); 

	free(pointList);
	free(vertList);
}
void Frame::printLists() {
	cout << "----------------------------------------------------------------------------------------------\n";
	cout << "POINT LIST: " << 2 * N << " elements\n{";
	for (int i = 0; i < 2 * N; i+=2) {
		if (i != 0) cout << ", ";
		cout << "(";
		cout << pointList[i]%1080 << ", " << pointList[i + 1]/1080;
		cout << ")";
	}
	cout << "}\n----------------------------------------------------------------------------------------------\n";
	cout << "VERTICE LIST: " << 2 * N << " elements\n{";
	for (int i = 0; i < 2 * N; i += 2) {
		if (i != 0) cout << ", ";
		cout << "(";
		cout << vertList[i] % 1080 << ", " << vertList[i + 1] / 1080;
		cout << ")";
	}
	cout << "}\n----------------------------------------------------------------------------------------------\n";
}
