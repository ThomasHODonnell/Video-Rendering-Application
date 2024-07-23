#include "Frame.cuh"
#include "Video.cuh"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <utility>
using namespace cv;
//using std::cout, std::endl, std::pair, std::tuple;
using namespace std;

#define N (1920*1080)
#define TPB 1024
const int GRIDSIZE = (N + TPB - 1) / TPB;
dim3 dimGrid = dim3(GRIDSIZE);
dim3 dimBlock = dim3(TPB);

Frame::Frame(Mat frame) {
	Vec3b firstPixel = frame.at<Vec3b>(Point(0, 0));
	processFrame(firstPixel);
}
__global__ void findPoints(Vec3b* in, pair<int, int>* out) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	int xCoord = i / 1080;
	int yCoord = i % 1080;
	Vec3b pixelColor = in[i];
	for (int j = -1081; j <= 1081; j++) {
		if (j == -1078) { j = -1; continue; }
		if (j == 1) { j = 1079; continue; }
		
		if (abs(in[i][0] - in[i + j][0]) < 50)
			if (abs(in[i][1] - in[i + j][1]) < 50)
				if (abs(in[i][2] - in[i + j][2]) < 50)
					continue;
		out[i] = { xCoord, yCoord };
	}
}
void Frame::processFrame(Vec3b firstPixel) {
	Vec3b* in = &firstPixel;
	pair<int,int>* out = 0;

	cudaError_t inMalErr = cudaMallocManaged(&in, N * sizeof(Vec3b));
	if (inMalErr != cudaSuccess)  printf("Input Array Malloc Error: code %d - %s.\n", cudaError(inMalErr), cudaGetErrorString(inMalErr)); return; 
	cudaError_t outMalErr = cudaMallocManaged(&out, N * sizeof(pair<int,int>));
	if (outMalErr != cudaSuccess) printf("Output Array Malloc Error: code %d - %s.\n", cudaError(outMalErr), cudaGetErrorString(outMalErr)); return;
	
	findPoints<<<dimGrid, dimBlock>>>(firstPixel);
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
