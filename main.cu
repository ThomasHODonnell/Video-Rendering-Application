#include <stdio.h>
#include <iostream>
#include <string>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include<glad/glad.h>
#include<GLFW/glfw3.h>

#include "auxiliaryFunc.cuh"

using namespace cv;
using namespace std;

const string path = "C:\\Users\\thoma\\source\\repos\\Video-Rendering-Application\\Videos\\Rubix.avi";

VideoCapture capture(path);
Mat frame;


const int FRAMES = capture.get(CAP_PROP_FRAME_COUNT);
const int ROWS = frame.rows;
const int COLUMNS = frame.cols;


int main(int argc, char** argv) { 
	
	if (!capture.isOpened()) {
		throw invalid_argument("File upload failed.");
		return -1;
	}

	namedWindow("Video Input", 1);
	bool firstFrame = true; 
	while(firstFrame) {
		capture >> frame;
		if (frame.empty())
			break;
		imshow("Video Input", frame);
		waitKey(20);
		printPrimatives(frame, capture);
		firstFrame = false;
	}
	waitKey(0);
	

	return 0;
}

