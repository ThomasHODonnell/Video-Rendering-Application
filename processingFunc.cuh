#include <stdio.h>
#include <iostream>
#include <string>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void prepFrame(Vec3b frameData, int* in, const int N);
