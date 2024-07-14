#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

void deviceProps() {
	int devices;
	cudaGetDeviceCount(&devices);
	printf("%d device(s) are connected in the system.\n", devices);
	for (int i = 0; i < devices; i++) {
		printf("----------------------------\n");
		cudaDeviceProp cdp;
		cudaGetDeviceProperties(&cdp, i);
		printf("Device Number: %d\n", i);
		printf("Device Name: %s\n", cdp.name);
		printf("Compute Capability: %d.%d\n", cdp.major, cdp.minor);
		printf("Max TPB: %d\n", cdp.maxThreadsPerBlock);
		printf("Shared Memory per Block: %lu bytes\n", cdp.reservedSharedMemPerBlock);
		printf("Total Global Memory: %lu bytes\n\n", cdp.totalGlobalMem);
	}
}

void videoProps(Mat frame, VideoCapture cap) {
	cout << "Frames: " << int(cap.get(CAP_PROP_FRAME_COUNT)) << endl;
	cout << "Rows: " << frame.rows << endl;
	cout << "Columns: " << int(frame.cols) << endl;
}

void printPrimatives(Mat frame, VideoCapture cap) {
	printf("----------------------------------------------------------------------------------------------\n\n");
	deviceProps();
	printf("----------------------------------------------------------------------------------------------\n\n");
	videoProps(frame, cap);
	printf("\n----------------------------------------------------------------------------------------------\n\n");
}