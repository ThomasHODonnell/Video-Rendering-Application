#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int* prepFrame(Vec3b frameData, int* in, const int N) {
	for (int i = 0; i < N; i += 3)
		for (int j = 0; j < 3; j++)
			in[i + j] = static_cast<int>(frameData[j]);
	return in;
}