#pragma once
#ifndef VIDEO_H
#define VIDEO_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

#include "auxiliaryFunc.cuh"

using namespace cv;
using std::vector;

class Video{
	public:
		int ROWS, COLUMNS, TYPE, FRAME_COUNT;
		vector<Vec3b*> sequence;
		
		Video();
		Video(Mat frame);

		void printVideoPrimatives();

		void incrementFC();

		void printPrimatives();

		void printSize();

		void updateSequence(Mat frame);

		Vec3b getFrame(int index);
};
#endif