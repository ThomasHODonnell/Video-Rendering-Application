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
__device__ int binarySearch(int* in, int left, int right, int errorX, int errorY, int refX, int refY) {
	if (right >= left) {
		int mid = ((left + right) / 2);
		if (mid % 2 == 0 && mid != right) mid++; // yCoord
		if (abs(in[mid] - refY) < errorY && abs(in[mid - 1] - refX) < errorX) {
			printf("INSIDE\n");
			printf("midY:%d, midX:%d, inY:%d, inX:%d\n", in[mid], in[mid - 1], refY, refX);
			float slopeMin = 0.1;
			float slope = abs(in[mid] - refY) / abs(in[mid - 1] - refX);
			if (slope >= slopeMin) return mid;
		}
		if (in[mid] < refY || (in[mid] == refY && in[mid - 1] < refX)) return binarySearch(in, left, mid - 1, errorX, errorY, refX, refY);
		else return binarySearch(in, mid + 1, right, errorX, errorY, refX, refY);
	}
	return -1;
}

__global__ void findVerts(int* in, int* out, int size, int width, int height) {
	const int errorY = 5, errorX = 5;
	const int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x); // xCoord
	if (i > size) return;
	if (i == 0) if (abs(in[i + 1] - in[i + 3]) > errorY) return; // drop isolated points if any
	if (i == size - 2) if (abs(in[i-1] - in[i+1]) > errorY) return;
	else if (abs(in[i+1] - in[i+3])>errorY || abs(in[i-1] - in[i+1]) > errorY) return;
	int mid = binarySearch(in, 0, size - 1, errorX, errorY, in[i], in[i + 1]);
	if (mid != -1) {
		out[i] = in[mid - 1]; 
		out[i + 1] = in[mid];
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

	for (int i = 0; i < N; i += 3) { // can this be done faster??
		data = frame.at<Vec3b>(Point(i % 1080, i / 1080));
		for (int j = 0; j < 3; j++) 
			inPL[i + j] = static_cast<int>(data[j]);
	}
	
	findPoints<<<dimGrid, dimBlock>>>(inPL, outPL, frame.cols, frame.rows);

	int* hostOutPL = (int*)calloc(3 * N, sizeof(int));
	cudaMemcpy(hostOutPL, outPL, N * sizeof(int), cudaMemcpyDeviceToHost);

	cudaError_t syncErrPL = cudaGetLastError();
	cudaError_t asyncErrPL = cudaDeviceSynchronize();
	if (syncErrPL != cudaSuccess) { printf("Sync Kernel Error: code %d - %s.\n", cudaError(syncErrPL), cudaGetErrorString(syncErrPL)); throw invalid_argument("Sync Kernel Error"); }
	if (asyncErrPL != cudaSuccess) { printf("Async Kernel Error: code %d - %s.\n", cudaError(asyncErrPL), cudaGetErrorString(asyncErrPL)); throw invalid_argument("Async Kernel Error"); }

	for (int i = 0; i < 3*N; i++) if (hostOutPL[i] != 0) pointList.push_back(hostOutPL[i]);
	
	cudaFree(inPL);
	cudaFree(outPL);

	int* inVL = 0;
	int* outVL = 0;

	cudaError_t inMalErrVL = cudaMallocManaged(&inVL, pointList.size() * sizeof(int));
	if (inMalErrVL != cudaSuccess) { printf("Input Array Malloc Error: code %d - %s.\n", cudaError(inMalErrVL), cudaGetErrorString(inMalErrVL)); return; }
	cudaError_t outMalErrVL = cudaMalloc(&outVL, pointList.size() * sizeof(int));
	if (outMalErrVL != cudaSuccess) { printf("Output Array Malloc Error: code %d - %s.\n", cudaError(outMalErrVL), cudaGetErrorString(outMalErrVL)); return; }
	cudaMemset(outVL, 0, pointList.size() * sizeof(int)); // overwrite inVL to prevent this O(n) copy? 

	//change pointlist vect to thrust::device_vector to prevent this copy? 
	for (int i = 0; i < pointList.size(); i++) inVL[i] = pointList.at(i); // O(n)

	findVerts << <(pointList.size() + 32 - 1) / 32, 32 >> > (inVL, outVL, pointList.size(), frame.cols, frame.rows);

	int* hostOutVL = (int*)calloc(pointList.size(), sizeof(int));
	cudaMemcpy(hostOutVL, outVL, pointList.size() * sizeof(int), cudaMemcpyDeviceToHost);

	cudaError_t syncErrVL = cudaGetLastError();
	cudaError_t asyncErrVL = cudaDeviceSynchronize();
 	if (syncErrVL != cudaSuccess) { printf("Sync Kernel Error: code %d - %s.\n", cudaError(syncErrVL), cudaGetErrorString(syncErrVL)); throw invalid_argument("Sync Kernel Error"); }
	if (asyncErrVL != cudaSuccess) { printf("Async Kernel Error: code %d - %s.\n", cudaError(asyncErrVL), cudaGetErrorString(asyncErrVL)); throw invalid_argument("Async Kernel Error"); }

	for (int i = 0; i < pointList.size(); i++) if (hostOutVL[i] != 0) vertList.push_back(hostOutVL[i]);

	cudaFree(inVL);
	cudaFree(outVL);
	
	printLists(); 
}
void Frame::printLists() {
	cout << "----------------------------------------------------------------------------------------------\n";
	cout << "POINT LIST: " << pointList.size() << " elements\n{";
	for (int i = 0; i < pointList.size(); i+=2) {
		if (i != 0) cout << ", ";
		cout << "(";
		cout << pointList.at(i)%1080 << ", " << pointList[i + 1]/1080;
		cout << ")";
	}
	cout << "}\n----------------------------------------------------------------------------------------------\n";
	cout << "VERTICE LIST: " << vertList.size() << " elements\n{";
	for (int i = 0; i < vertList.size(); i += 2) {
		if (i != 0) cout << ", ";
		cout << "(";
		cout << vertList.at(i) % 1080 << ", " << vertList[i + 1] / 1080;
		cout << ")";
	}
	cout << "}\n----------------------------------------------------------------------------------------------\n";
}
