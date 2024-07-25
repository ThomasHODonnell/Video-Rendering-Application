#include <stdio.h>
#include <iostream>
#include <string>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include<glad/glad.h>
#include<GLFW/glfw3.h>

#include "Video.cuh"
#include "Frame.cuh"
#include "auxiliaryFunc.cuh"
#include "processingFunc.cuh"

using namespace cv;
using std::cout;
using std::endl;

const string path = "C:\\Users\\thoma\\source\\repos\\Video-Rendering-Application\\Videos\\Rubix.avi";


VideoCapture capture(path);
Mat frame;

__global__ void testK() {};

int main(int argc, char** argv) { 
	
	if (!capture.isOpened()) {
		throw invalid_argument("File upload failed.");
		return -1;
	}

	capture >> frame;

	Video V(frame);

	namedWindow("Video Input", 1);
	bool first = true;
    while (first) {

		capture >> frame;
		
        if (frame.empty())
            break;

		imshow("Video Input", frame);
		
		V.updateSequence(frame);
		Frame* current = new Frame(frame);

        // Press 'q' to exit the loop
        if (waitKey(30) >= 0)
            break;
		first = false;
    }
	waitKey(0);
	capture.release();
	destroyAllWindows();

	return 0;
}

