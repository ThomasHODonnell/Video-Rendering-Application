#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <opencv2/opencv.hpp>

#include "Video.cuh"

using namespace cv;
using namespace std;

void deviceProps();