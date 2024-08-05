#pragma once
#ifndef FRAME_CUH
#define FRAME_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include<GLFW/glfw3.h>
#include <vector>
using cv::Mat;
using namespace std;
class Frame {
	public:
		int* pointList;
		int* vertList;

		Frame();
		void processFrame(Mat frame);
		GLfloat* getVerts();
		float domainFlip(int alpha, int max, int min);
		void printLists();
};
	
#endif