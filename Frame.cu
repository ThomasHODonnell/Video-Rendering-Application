#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <stdlib.h>
#include "Frame.cuh"
#include "Video.cuh"

using namespace cv;
using namespace std;

const int N = (1920 * 1080), TPB = 1024;
const int GRIDSIZE = (N + TPB - 1) / TPB;
dim3 dimGrid = dim3(GRIDSIZE);
dim3 dimBlock = dim3(TPB);

int MAXPT = 0, MINPT = -1; 

Frame::Frame() {}

__global__ void findPoints(int* in, int* out, int width, int height) {
	const int i = 3 * (blockIdx.x * blockDim.x + threadIdx.x);
	int xCoord = (i / 3) % width;
	int yCoord = (i / 3) / width;
	if (xCoord == 0 || xCoord == width - 1 || yCoord == 0 || yCoord == height - 1) return;
	//MAXPT = 5; 

	for (int j = -1; j <= 1; j++) {
		for (int k = -1; k <= 1; k++) {
			if (j == 0 && k == 0) continue;
			int neighborIdx = i + 3 * (j * width + k);
			if (neighborIdx >= 0 && neighborIdx < width * height * 3)
				if (abs(in[neighborIdx] - in[i]) >= 5 ||
					abs(in[neighborIdx + 1] - in[i + 1]) >= 5 ||
					abs(in[neighborIdx + 2] - in[i + 2]) >= 5)
					out[(i/3)*2] = 1; // 0 - 2*N-1 indexed, return more information? 
				else out[(i/3)*2] = 0; 
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

	cudaError_t inMalErrPL = cudaMallocManaged(&inPL, 3 * N * sizeof(int)); // RGB * N
	if (inMalErrPL != cudaSuccess) { printf("Input Array Malloc Error: code %d - %s.\n", cudaError(inMalErrPL), cudaGetErrorString(inMalErrPL)); return; }
	cudaError_t outMalErrPL = cudaMallocManaged(&pointList, 2 * N * sizeof(int)); // X, Y * N
	if (outMalErrPL != cudaSuccess) { printf("Output Array Malloc Error: code %d - %s.\n", cudaError(outMalErrPL), cudaGetErrorString(outMalErrPL)); return; }

	for (int i = 0; i < N; i += 3) { // can this be done faster??
		data = frame.at<Vec3b>(Point(i % 1080, i / 1080));
		for (int j = 0; j < 3; j++) 
			inPL[i + j] = static_cast<int>(data[j]);
	}
	
	findPoints<<<dimGrid, dimBlock>>>(inPL, pointList, frame.cols,  frame.rows);

	cudaError_t syncErrPL = cudaGetLastError();
	cudaError_t asyncErrPL = cudaDeviceSynchronize();
	if (syncErrPL != cudaSuccess) { printf("Sync Kernel Error: code %d - %s.\n", cudaError(syncErrPL), cudaGetErrorString(syncErrPL)); throw invalid_argument("Sync Kernel Error"); }
	if (asyncErrPL != cudaSuccess) { printf("Async Kernel Error: code %d - %s.\n", cudaError(asyncErrPL), cudaGetErrorString(asyncErrPL)); throw invalid_argument("Async Kernel Error"); }

	cudaFree(inPL);

	cudaError_t vlMalErr = cudaMalloc(&vertList, 2 * N * sizeof(int));
	if (vlMalErr != cudaSuccess) { printf("Output Array Malloc Error: code %d - %s.\n", cudaError(vlMalErr), cudaGetErrorString(vlMalErr)); return; }

	findVerts << <(2 * N + TPB - 1) / TPB, TPB >> > (pointList, vertList, 2 * N);

	cudaError_t syncErrvertList = cudaGetLastError();
	cudaError_t asyncErrvertList = cudaDeviceSynchronize();
 	if (syncErrvertList != cudaSuccess) { printf("Sync Kernel Error: code %d - %s.\n", cudaError(syncErrvertList), cudaGetErrorString(syncErrvertList)); throw invalid_argument("Sync Kernel Error"); }
	if (asyncErrvertList != cudaSuccess) { printf("Async Kernel Error: code %d - %s.\n", cudaError(asyncErrvertList), cudaGetErrorString(asyncErrvertList)); throw invalid_argument("Async Kernel Error"); }

	getVerts(); 
	
	Mat output(frame.rows, frame.cols, CV_8UC3, Scalar(0, 0, 0));

	Vec3b yel = { 255, 255, 0 };
	int x, y; 
	for (int i = 0; i < 2 * N; i += 2) {
		x = (i/2 % frame.cols) * 3;
		y = (i/2 / frame.cols) * 2 * 1.33;  
		if (pointList[i] != 0 && x < frame.cols && y < frame.rows)
			output.at<Vec3b>(Point(x, y)) = yel;
	}

	imshow("PL Output", output);
	
}
float Frame::domainFlip(int alpha, int max, int min) { return 2 * ((alpha - min) / (max - min)) - 1; }

GLfloat* Frame::getVerts() {
	int maxPos = 0, minPos = 2*N, size = 0; 
	vector<int> points; 
	for (int i = 0; i < 2 * N; i += 2) {
		if (pointList[i] != 0) {
			if (i > maxPos) maxPos = i; 
			if (i < minPos) minPos = i; 
			size++;
			points.push_back(i);
		}
	}
	GLfloat* verts = (GLfloat*)malloc(size * 3 * sizeof(GLfloat));
	int index = 0; 
	for (int i = 0; i < points.size() ; i++, index++) {
			verts[index] = domainFlip(points.at(i) % 1080, maxPos, minPos);
			verts[++index] = domainFlip(points.at(i) / 1080, maxPos, minPos);
			verts[++index] = 0.0f;
	}
	return verts; 
}
void Frame::printLists() {
	cout << "----------------------------------------------------------------------------------------------\n";
	cout << "POINT LIST: " << 2 * N << " elements\n{";
	for (int i = 0; i < N; i+=2) {
		if (i != 0) cout << ", ";
		cout << "(";
		cout << pointList[i]%1080 << ", " << pointList[i + 1]/1080;
		cout << ")";
	}
	cout << "}\n----------------------------------------------------------------------------------------------\n";
	cout << "VERTICE LIST: " << 2 * N << " elements\n{";
	for (int i = 0; i < N; i += 2) {
		if (i != 0) cout << ", ";
		cout << "(";
		cout << vertList[i] % 1080 << ", " << vertList[i + 1] / 1080;
		cout << ")";
	}
	cout << "}\n----------------------------------------------------------------------------------------------\n";
}
