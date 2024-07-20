#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <opencv2/opencv.hpp>

#include <stdio.h>
#include <iostream>

#include "Video.h"

using namespace cv;
using namespace std;

void deviceProps() {
	int devices;
	cudaGetDeviceCount(&devices);
	printf("%d device(s) are connected in the system.\n\n", devices);
	for (int i = 0; i < devices; i++) {
		printf("----------------------------------------------------------------------------------------------\n");
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