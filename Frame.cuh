#pragma once
#ifndef FRAME_CUH
#define FRAME_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <vector>
using cv::Mat;
using namespace std;
class Frame {
	public:
		int* pointList;
		int* vertList;

		Frame(Mat frame);
		void processFrame(Mat frame);
		void printLists();
};
	
#endif