#include "Frame.cuh"
//#include "Video.cuh"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <utility>
using namespace cv;
//using std::cout, std::endl, std::pair;
using namespace std;

Frame::Frame(Mat frame) {
	Vec3b firstPixel = frame.at<Vec3b>(Point(0, 0));
	fillLists(firstPixel, {0,0});
}
void Frame::fillLists(Vec3b firstPixel, pair<int,int> coord) {
	pointList.push_back({ coord.first, coord.second });
	printLists();
}
void Frame::printLists() {
	cout << "----------------------------------------------------------------------------------------------\n";
	cout << "{ ";
	for (int i = 0; i < pointList.size(); i++) {
		if (i != 0) cout << ", ";
		cout << "(";
		for (int j = 0; j < pointList[i].size(); j++) {
			cout << pointList[i][j];
			if (j < pointList[i].size() - 1) cout << ", ";
		}
		cout << ")";
	}
	cout << " }\n----------------------------------------------------------------------------------------------\n";
	cout << "{ ";
	for (int i = 0; i < vertList.size(); i++) {
		if (i != 0) cout << ", ";
		cout << "(";
		for (int j = 0; j < vertList[i].size(); j++) {
			cout << vertList[i][j];
			if (j < vertList[i].size() - 1) cout << ", ";
		}
		cout << ")";
	}
	cout << "----------------------------------------------------------------------------------------------\n";
}
