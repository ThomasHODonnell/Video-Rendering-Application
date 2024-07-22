#pragma once
#ifndef FRAME_CUH
#define FRAME_CUH

#include <opencv2/opencv.hpp>
#include <vector>
using namespace cv;
//using std::vector, std::pair;
using namespace std;
class Frame {
	public:
		vector <vector<int>> pointList;
		vector <vector<int>> vertList;

		Frame(Mat frame);
		void fillLists(Vec3b firstPixel, pair<int,int> coord);
		void printLists(); 
};

#endif