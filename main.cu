#include <stdio.h>
#include <string>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include<glad/glad.h>
#include<GLFW/glfw3.h>

using namespace cv;
using namespace std;

const string path = "C:\\Users\\thoma\\source\\repos\\Video-Rendering-Application\\Videos\\Rubix.avi";


int main(int argc, char** argv) { 
	
	Mat frame;
	VideoCapture capture(path);
	
	if (!capture.isOpened()) {
		throw invalid_argument("File upload failed.");
		return -1;
	}

	namedWindow("Video Input", 1);
	for (;;)
	{
		capture >> frame;
		if (frame.empty())
			break;
		imshow("Video Input", frame);
		waitKey(20);
	}
	waitKey(0);
	
	return 0;
}

