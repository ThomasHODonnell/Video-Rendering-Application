#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "Video.h"
#include "auxiliaryFunc.cuh"

using namespace cv;
using namespace std;

Video::Video() {};
Video::Video(Mat frame) {
		ROWS = frame.rows;
		COLUMNS = frame.cols;
		TYPE = frame.type();
		FRAME_COUNT = 0;
		printPrimatives();
	}
void Video::printVideoPrimatives() {
		cout << "Rows: " << ROWS;
		cout << "\nColumns: " << COLUMNS;
		cout << "\nFrame Type: " << TYPE;
	}
void Video::incrementFC() {
		FRAME_COUNT++;
	}

void Video::printPrimatives() {
		cout << "----------------------------------------------------------------------------------------------\n\n";
		deviceProps();
		cout << "----------------------------------------------------------------------------------------------\n\n";
		printVideoPrimatives();
		cout << "\n\n----------------------------------------------------------------------------------------------\n\n";
}
void Video::printSize() { cout << sequence.size(); }
void Video::updateSequence(Mat frame) {
	Vec3b color = frame.at<Vec3b>(Point(0, 0));
	Vec3b* pixelPtr = &color;
	sequence.push_back(pixelPtr);
	incrementFC();
}
Vec3b Video::getFrame(int index) {
	return *sequence.at(index);
}




