#pragma once
#ifndef FRAME_CUH
#define FRAME_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace cv;
using namespace std;
class Frame {
	public:
		vector <vector<int>> pointList;
		vector <vector<int>> vertList;

		Frame(Mat frame);
		void processFrame(Vec3b* data);
		void printLists(); 
};

#endif